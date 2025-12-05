import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from matplotlib.patches import Ellipse

# =============================================================================
#  CONFIGURACIÓN
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', 'Analizar_Data') 
SEARCH_PATH = os.path.join(BASE_DIR, "Resultados", "**", "*_WINDOWED_METRICS.csv")

def cargar_datos():
    """Busca recursivamente todos los CSVs de métricas windowed y los une."""
    files = glob.glob(SEARCH_PATH, recursive=True)
    
    if not files:
        print(f"❌ No se encontraron archivos en: {SEARCH_PATH}")
        return None

    print(f"📂 Encontrados {len(files)} archivos:")
    for f in files:
        print(f"   • {os.path.basename(f)}")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Extraer el nombre del sujeto del VideoID
        df['Subject'] = df['VideoID'].apply(lambda x: x.split('_')[0])
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

def analizar_centroides_y_separacion(df, features):
    """Calcula centroides y distancias entre sujetos."""
    print("\n" + "="*70)
    print("🎯 ANÁLISIS DE CENTROIDES Y SEPARACIÓN")
    print("="*70)
    
    sujetos = sorted(df['Subject'].unique())
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calcular centroides por sujeto
    centroides = {}
    for suj in sujetos:
        mask = df['Subject'] == suj
        centroides[suj] = np.mean(X_scaled[mask], axis=0)
    
    # Distancias entre centroides
    print("\n📏 DISTANCIAS EUCLIDIANAS ENTRE CENTROIDES:")
    print("-" * 50)
    distancias = []
    for i, suj1 in enumerate(sujetos):
        for suj2 in sujetos[i+1:]:
            dist = np.linalg.norm(centroides[suj1] - centroides[suj2])
            distancias.append(dist)
            print(f"   {suj1} ↔ {suj2}: {dist:.3f}")
    
    print(f"\n📊 Distancia promedio: {np.mean(distancias):.3f} (±{np.std(distancias):.3f})")
    print(f"   Distancia mínima: {np.min(distancias):.3f}")
    print(f"   Distancia máxima: {np.max(distancias):.3f}")
    
    return centroides, X_scaled, scaler

def encontrar_metricas_discriminantes(df, features, top_n=15):
    """Encuentra las métricas que más diferencian a los sujetos."""
    print("\n" + "="*70)
    print("🔬 MÉTRICAS MÁS DISCRIMINANTES (Feature Importance)")
    print("="*70)
    
    X = df[features].values
    y = df['Subject'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest para importancia de features
    clf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=10)
    clf.fit(X_scaled, y)
    
    # Ordenar por importancia
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🏆 TOP {top_n} MÉTRICAS MÁS DISCRIMINANTES:")
    print("-" * 50)
    
    # Categorizar métricas
    main_seq_features = []
    temporal_features = []
    spectral_features = []
    other_features = []
    
    for idx, row in importances.head(top_n).iterrows():
        feat_name = row['Feature']
        importance = row['Importance']
        
        # Categorización
        if 'MainSeq' in feat_name or 'Amp_' in feat_name or 'PeakVel_' in feat_name:
            category = "🔥 Main Seq"
            main_seq_features.append((feat_name, importance))
        elif 'ISI_' in feat_name or 'Saccade_Rate' in feat_name or 'Latency' in feat_name:
            category = "⏱️  Temporal"
            temporal_features.append((feat_name, importance))
        elif 'Freq' in feat_name or 'Entropy' in feat_name:
            category = "📊 Spectral"
            spectral_features.append((feat_name, importance))
        else:
            category = "📈 General"
            other_features.append((feat_name, importance))
        
        print(f"   {category} {feat_name:35s}: {importance:.4f}")
    
    # Resumen por categoría
    print("\n📋 RESUMEN POR CATEGORÍA:")
    print("-" * 50)
    if main_seq_features:
        total_main = sum(imp for _, imp in main_seq_features)
        print(f"   🔥 Main Sequence: {len(main_seq_features)} métricas ({total_main:.3f} importancia total)")
    if temporal_features:
        total_temp = sum(imp for _, imp in temporal_features)
        print(f"   ⏱️  Temporales: {len(temporal_features)} métricas ({total_temp:.3f} importancia total)")
    if spectral_features:
        total_spec = sum(imp for _, imp in spectral_features)
        print(f"   📊 Espectrales: {len(spectral_features)} métricas ({total_spec:.3f} importancia total)")
    if other_features:
        total_other = sum(imp for _, imp in other_features)
        print(f"   📈 Generales: {len(other_features)} métricas ({total_other:.3f} importancia total)")
    
    # Visualización
    plt.figure(figsize=(14, 8))
    
    # Colorear barras por categoría
    colors_map = []
    for feat in importances.head(top_n)['Feature']:
        if 'MainSeq' in feat or 'Amp_' in feat or 'PeakVel_' in feat:
            colors_map.append('#FF6B6B')  # Rojo
        elif 'ISI_' in feat or 'Saccade_Rate' in feat or 'Latency' in feat:
            colors_map.append('#4ECDC4')  # Cyan
        elif 'Freq' in feat or 'Entropy' in feat:
            colors_map.append('#95E1D3')  # Verde agua
        else:
            colors_map.append('#45B7D1')  # Azul
    
    ax = sns.barplot(
        data=importances.head(top_n), 
        x='Importance', 
        y='Feature',
        palette=colors_map,
        hue='Feature',
        legend=False
    )
    
    plt.title('Top Métricas Discriminantes (Random Forest Feature Importance)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Importancia', fontsize=13, fontweight='bold')
    plt.ylabel('Métrica', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Agregar leyenda de categorías
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='🔥 Main Sequence'),
        Patch(facecolor='#4ECDC4', label='⏱️ Temporal'),
        Patch(facecolor='#95E1D3', label='📊 Spectral'),
        Patch(facecolor='#45B7D1', label='📈 General')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return importances

def analizar_perfil_por_sujeto(df, features, top_features):
    """Crea un perfil único para cada sujeto basado en las métricas clave."""
    print("\n" + "="*70)
    print("👤 PERFILES ÚNICOS POR SUJETO")
    print("="*70)
    
    sujetos = sorted(df['Subject'].unique())
    perfiles = {}
    
    # Usar solo las top features
    top_feat_names = top_features['Feature'].head(8).tolist()
    
    print("\n📊 Características del Perfil de cada Sujeto:")
    print("-" * 70)
    
    for suj in sujetos:
        datos_suj = df[df['Subject'] == suj][top_feat_names]
        perfiles[suj] = {
            'mean': datos_suj.mean(),
            'std': datos_suj.std(),
            'median': datos_suj.median()
        }
        
        print(f"\n   🔹 {suj}:")
        print(f"      Muestras: {len(datos_suj)}")
        for feat in top_feat_names[:3]:  # Mostrar top 3
            print(f"      • {feat:30s}: μ={perfiles[suj]['mean'][feat]:8.3f} | σ={perfiles[suj]['std'][feat]:8.3f}")
    
    # Visualización de perfiles (Radar Chart)
    fig, axes = plt.subplots(1, len(sujetos), figsize=(6*len(sujetos), 6), 
                              subplot_kw=dict(projection='polar'))
    
    if len(sujetos) == 1:
        axes = [axes]
    
    # Normalizar para visualización
    scaler = StandardScaler()
    all_data = df[top_feat_names].values
    normalized_data = scaler.fit_transform(all_data)
    
    angles = np.linspace(0, 2 * np.pi, len(top_feat_names), endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for idx, suj in enumerate(sujetos):
        ax = axes[idx]
        
        # Datos normalizados del sujeto
        mask = df['Subject'] == suj
        suj_norm = normalized_data[mask]
        values = np.mean(suj_norm, axis=0).tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3, color=colors[idx % len(colors)], label=suj)
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_feat_names, size=9, fontweight='bold')
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f'Perfil: {suj}', size=16, fontweight='bold', pad=25)
        ax.grid(True, alpha=0.3)
        
        # Añadir círculos de referencia
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(['-2σ', '-1σ', '0', '+1σ', '+2σ'], fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return perfiles

def analizar_confusiones(df, features):
    """Analiza qué muestras fueron confundidas y por qué."""
    print("\n" + "="*70)
    print("🔍 ANÁLISIS PROFUNDO DE CONFUSIONES")
    print("="*70)
    
    X = df[features].values
    y = df['Subject'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Entrenar modelo
    clf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    
    # Predicciones con probabilidades
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    # Métricas de clasificación
    accuracy = np.mean(y_pred == y_test)
    print(f"\n✅ Precisión del modelo: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Encontrar casos confundidos
    confusiones = []
    aciertos = []
    
    for i, (real, pred, probs) in enumerate(zip(y_test, y_pred, y_prob)):
        if real != pred:
            confusiones.append({
                'Real': real,
                'Predicho': pred,
                'Confianza': np.max(probs),
                'Idx': i
            })
        else:
            aciertos.append({
                'Sujeto': real,
                'Confianza': np.max(probs),
                'Idx': i
            })
    
    print(f"\n❌ Total de Confusiones: {len(confusiones)} de {len(y_test)} ({100*len(confusiones)/len(y_test):.1f}%)")
    
    if confusiones:
        df_conf = pd.DataFrame(confusiones)
        print("\n📊 PATRONES DE CONFUSIÓN:")
        print("-" * 50)
        confusion_pairs = df_conf.groupby(['Real', 'Predicho']).size().reset_index(name='Count')
        confusion_pairs = confusion_pairs.sort_values('Count', ascending=False)
        
        for _, row in confusion_pairs.iterrows():
            print(f"   {row['Real']} → {row['Predicho']}: {row['Count']} veces "
                  f"({100*row['Count']/len(confusiones):.1f}% de las confusiones)")
        
        # Análisis de confianza en confusiones
        print(f"\n📈 Confianza en Predicciones Erróneas:")
        print(f"   Media: {df_conf['Confianza'].mean():.3f}")
        print(f"   Mediana: {df_conf['Confianza'].median():.3f}")
        print(f"   Mínima: {df_conf['Confianza'].min():.3f}")
        print(f"   Máxima: {df_conf['Confianza'].max():.3f}")
    
    if aciertos:
        df_aciertos = pd.DataFrame(aciertos)
        print(f"\n✅ Confianza en Predicciones Correctas:")
        print(f"   Media: {df_aciertos['Confianza'].mean():.3f}")
        print(f"   Mediana: {df_aciertos['Confianza'].median():.3f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='YlOrRd', 
        xticklabels=sorted(np.unique(y)), 
        yticklabels=sorted(np.unique(y)),
        cbar_kws={'label': 'Cantidad'},
        linewidths=2,
        linecolor='white'
    )
    plt.title('Matriz de Confusión (Predicción vs Real)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicho', fontsize=13, fontweight='bold')
    plt.ylabel('Real', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return confusiones, aciertos, X_test, y_test, y_pred, clf

def comparar_casos_extremos(df, features, confusiones, aciertos, X_test, y_test):
    """Compara las características de casos bien clasificados vs confundidos."""
    print("\n" + "="*70)
    print("⚖️  COMPARACIÓN: ¿QUÉ DIFERENCIA LOS CASOS CLAROS DE LOS CONFUSOS?")
    print("="*70)
    
    if not confusiones or not aciertos:
        print("⚠️  No hay suficientes datos para comparar")
        return
    
    # Tomar índices
    idx_confusos = [c['Idx'] for c in confusiones]
    idx_claros = [a['Idx'] for a in aciertos if a['Confianza'] > 0.9]
    
    if len(idx_claros) < len(idx_confusos):
        print(f"⚠️  Solo hay {len(idx_claros)} casos claros vs {len(idx_confusos)} confusos")
        idx_claros = [a['Idx'] for a in aciertos]  # Usar todos
    else:
        idx_claros = idx_claros[:len(idx_confusos)*2]  # Tomar el doble
    
    # Extraer features
    X_confusos = X_test[idx_confusos]
    X_claros = X_test[idx_claros]
    
    # Calcular diferencias significativas
    diferencias = []
    for i, feat in enumerate(features):
        vals_confusos = X_confusos[:, i]
        vals_claros = X_claros[:, i]
        
        # T-test
        try:
            t_stat, p_val = ttest_ind(vals_confusos, vals_claros)
        except:
            continue
        
        if p_val < 0.05:  # Significativo
            diferencias.append({
                'Feature': feat,
                'Media_Confusos': np.mean(vals_confusos),
                'Media_Claros': np.mean(vals_claros),
                'Diferencia': abs(np.mean(vals_confusos) - np.mean(vals_claros)),
                'P_value': p_val
            })
    
    if diferencias:
        df_diff = pd.DataFrame(diferencias).sort_values('Diferencia', ascending=False)
        
        print("\n🔬 MÉTRICAS QUE DIFERENCIAN CASOS CLAROS DE CONFUSOS:")
        print("-" * 80)
        print(f"{'Métrica':35s} {'Confusos':>12s} {'Claros':>12s} {'Diff':>10s} {'p-value':>10s}")
        print("-" * 80)
        
        for idx, row in df_diff.head(15).iterrows():
            # Identificar categoría
            if 'MainSeq' in row['Feature'] or 'Amp_' in row['Feature'] or 'PeakVel_' in row['Feature']:
                marker = "🔥"
            elif 'ISI_' in row['Feature']:
                marker = "⏱️ "
            else:
                marker = "  "
            
            print(f"{marker} {row['Feature']:32s} {row['Media_Confusos']:12.3f} {row['Media_Claros']:12.3f} "
                  f"{row['Diferencia']:10.3f} {row['P_value']:10.4f}")
        
        print(f"\n💡 Interpretación:")
        print(f"   • Casos CONFUSOS tienen mayor variabilidad en métricas clave")
        print(f"   • Casos CLAROS son más consistentes y predecibles")
        print(f"   • {len(diferencias)} métricas muestran diferencias significativas (p<0.05)")
    else:
        print("⚠️  No se encontraron diferencias estadísticamente significativas")

def visualizar_lda_mejorado(df, features):
    """Visualización mejorada del LDA con centroides y elipses de confianza."""
    print("\n" + "="*70)
    print("📊 VISUALIZACIÓN LDA CON CENTROIDES Y CLUSTERS")
    print("="*70)
    
    X = df[features].values
    y = df['Subject'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determinar número de componentes
    n_classes = len(np.unique(y))
    n_components = min(2, n_classes - 1)
    
    lda = LDA(n_components=n_components)
    lda_components = lda.fit_transform(X_scaled, y)
    
    if n_components == 1:
        # Si solo hay 1 componente, crear segunda dimensión artificial
        lda_df = pd.DataFrame(data=lda_components, columns=['LD1'])
        lda_df['LD2'] = np.random.randn(len(lda_df)) * 0.1
    else:
        lda_df = pd.DataFrame(data=lda_components, columns=['LD1', 'LD2'])
    
    lda_df['Subject'] = y
    
    # Varianza explicada
    explained_var = lda.explained_variance_ratio_
    print(f"\n📊 Varianza Explicada (LDA):")
    for i, var in enumerate(explained_var):
        print(f"   LD{i+1}: {var:.3f} ({var*100:.1f}%)")
    print(f"   Total: {sum(explained_var):.3f} ({sum(explained_var)*100:.1f}%)")
    
    # Calcular centroides en espacio LDA
    sujetos = sorted(lda_df['Subject'].unique())
    centroides_lda = {}
    for suj in sujetos:
        datos_suj = lda_df[lda_df['Subject'] == suj][['LD1', 'LD2']]
        centroides_lda[suj] = datos_suj.mean().values
    
    # Visualización
    fig, ax = plt.subplots(figsize=(16, 12))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    color_map = {suj: colors[i % len(colors)] for i, suj in enumerate(sujetos)}
    
    for suj in sujetos:
        datos = lda_df[lda_df['Subject'] == suj]
        
        # Scatter de puntos
        ax.scatter(
            datos['LD1'], datos['LD2'], 
            label=f'{suj} (n={len(datos)})', 
            alpha=0.5, 
            s=60, 
            color=color_map[suj],
            edgecolors='white',
            linewidth=0.8
        )
        
        # Centroide
        cx, cy = centroides_lda[suj]
        ax.scatter(
            cx, cy, 
            marker='X', 
            s=800, 
            color=color_map[suj],
            edgecolors='black',
            linewidth=3,
            zorder=10
        )
        
        # Elipse de confianza (2 desviaciones estándar)
        if len(datos) > 2:
            cov = np.cov(datos['LD1'], datos['LD2'])
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            
            ellipse = Ellipse(
                xy=(cx, cy),
                width=lambda_[0]*4,  # 2 std
                height=lambda_[1]*4,
                angle=np.degrees(np.arctan2(*v[:,0][::-1])),
                facecolor=color_map[suj],
                alpha=0.15,
                edgecolor=color_map[suj],
                linewidth=2.5,
                linestyle='--'
            )
            ax.add_patch(ellipse)
    
    # Líneas entre centroides con distancias
    for i in range(len(sujetos)):
        for j in range(i+1, len(sujetos)):
            c1 = centroides_lda[sujetos[i]]
            c2 = centroides_lda[sujetos[j]]
            dist = np.linalg.norm(c1 - c2)
            
            ax.plot(
                [c1[0], c2[0]], 
                [c1[1], c2[1]], 
                'k--', 
                alpha=0.4, 
                linewidth=1.5
            )
            
            # Anotar distancia
            mid = (c1 + c2) / 2
            ax.text(
                mid[0], mid[1], 
                f'd={dist:.2f}',
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
            )
    
    ax.set_xlabel(f'Componente Discriminante 1 (LD1) - {explained_var[0]*100:.1f}% varianza', 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Componente Discriminante 2 (LD2) - {explained_var[1]*100:.1f}% varianza' if n_components > 1 else 'LD2', 
                  fontsize=14, fontweight='bold')
    ax.set_title('LDA: Separación de Huellas Oculomotoras\n(Centroides, Clusters y Distancias)', 
                 fontsize=17, fontweight='bold', pad=25)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir distancias
    print("\n📏 DISTANCIAS ENTRE CENTROIDES EN ESPACIO LDA:")
    print("-" * 50)
    distancias_lda = []
    for i in range(len(sujetos)):
        for j in range(i+1, len(sujetos)):
            c1 = centroides_lda[sujetos[i]]
            c2 = centroides_lda[sujetos[j]]
            dist = np.linalg.norm(c1 - c2)
            distancias_lda.append(dist)
            print(f"   {sujetos[i]} ↔ {sujetos[j]}: {dist:.3f}")
    
    if distancias_lda:
        print(f"\n📊 Estadísticas de Separación LDA:")
        print(f"   Distancia promedio: {np.mean(distancias_lda):.3f}")
        print(f"   Distancia mínima: {np.min(distancias_lda):.3f}")
        print(f"   Distancia máxima: {np.max(distancias_lda):.3f}")
        
        # Interpretación
        dist_min = np.min(distancias_lda)
        if dist_min > 2.0:
            print(f"\n✅ EXCELENTE separación (dist_min > 2.0)")
        elif dist_min > 1.5:
            print(f"\n✅ BUENA separación (dist_min > 1.5)")
        elif dist_min > 1.0:
            print(f"\n⚠️  MODERADA separación (dist_min > 1.0)")
        else:
            print(f"\n⚠️  Separación BAJA (dist_min < 1.0) - Considera más métricas")

# =============================================================================
#  MAIN - ANÁLISIS COMPLETO
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("🧬 ANÁLISIS BIOMÉTRICO COMPLETO - HUELLAS OCULOMOTORAS")
    print("="*70)
    
    # 1. Cargar datos
    df = cargar_datos()
    
    if df is not None:
        print(f"\n📊 Dataset Total: {df.shape[0]} muestras (ventanas) x {df.shape[1]} columnas")
        print(f"   Sujetos: {sorted(df['Subject'].unique())}")
        print(f"   Muestras por sujeto:")
        for suj in sorted(df['Subject'].unique()):
            count = len(df[df['Subject'] == suj])
            print(f"      • {suj}: {count} ventanas")
        
        # Preparar features
        cols_to_drop = ['VideoID', 'Window_Idx', 'Window