import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# Importar toolkit 3D
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns
import os
import glob
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings

# Intentar configurar el backend interactivo para los gráficos 3D
try:
    import matplotlib
    matplotlib.use('TkAgg') 
except:
    pass 

warnings.filterwarnings('ignore')

# =============================================================================
#  CONFIGURACIÓN Y RUTAS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
# Ajusta esta ruta si tu carpeta de entrada es distinta
INPUT_PATH = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_Avanzados", "Reportes_Finales_Interactivos", f"Analisis_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo visual
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# =============================================================================
#  DICCIONARIO DE TRADUCCIÓN (INGLÉS -> ESPAÑOL)
# =============================================================================
# Agrega aquí cualquier otra métrica que aparezca en tu CSV
DICCIONARIO_METRICAS = {
    # --- Pupila ---
    'Pupil_Mean': 'Diámetro Pupilar Prom.',
    'Pupil_Vel_Max': 'Vel. Máx. Pupila',
    'Pupil_Std': 'Variabilidad Pupilar',
    'Pupil_Vel_Mean': 'Vel. Prom. Pupila',
    
    # --- Movimiento Ocular (Sácadas y Fijaciones) ---
    'Microsaccade_Rate': 'Tasa Microsacadas',
    'Velocity_Transition_Smoothness': 'Suavidad Transición',
    'Jerk_Mean': 'Jerk (Sacudida) Prom.',
    'Main_Seq_Slope': 'Pendiente Secuencia Princ.',
    'Vel_Mean': 'Velocidad Ocular Prom.',
    'Acc_Max': 'Aceleración Máx.',
    'Fixation_Vel_Mean': 'Vel. en Fijaciones',
    
    # --- Complejidad y Entropía ---
    'Fractal_Dim': 'Dimensión Fractal',
    'Spatial_Entropy': 'Entropía Espacial',
    'Lempel_Ziv': 'Complejidad Lempel-Ziv',
    
    # --- Otros ---
    'Saccade_Duration_Mean': 'Duración Sácadas Prom.',
    'Fixation_Duration_Mean': 'Duración Fijación Prom.',
    'Blink_Rate': 'Tasa de Parpadeo'
}

def traducir(texto):
    """Devuelve la traducción o el texto original si no se encuentra."""
    return DICCIONARIO_METRICAS.get(texto, texto.replace('_', ' '))

# =============================================================================
#  HELPER: ELIPSES DE CONFIANZA (Solo para 2D)
# =============================================================================
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# =============================================================================
#  1. CARGA DE DATOS
# =============================================================================
def cargar_dataset():
    print("="*70)
    print("🚀 ANÁLISIS BIOMÉTRICO: TRADUCIDO Y AJUSTADO")
    print("="*70)
    
    files = glob.glob(INPUT_PATH, recursive=True)
    if not files: 
        print("❌ No se encontraron archivos CSV.")
        return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except: pass
            
    if not dfs: return None
    df_final = pd.concat(dfs, ignore_index=True)
    df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Filtro básico: asegurar que hay datos de pupila válidos
    if 'Pupil_Mean' in df_final.columns:
        df_final = df_final[df_final['Pupil_Mean'] > 0]
    
    print(f"✅ Dataset cargado: {len(df_final)} muestras.")
    return df_final

# =============================================================================
#  2. SELECCIÓN DE CARACTERÍSTICAS (RANDOM FOREST) - TRADUCIDO
# =============================================================================
def analizar_importancia_features(df_in):
    print("\n" + "-"*30 + " RANKING DE IMPORTANCIA (RANDOM FOREST) " + "-"*30)
    
    # 1. Preparar datos
    X = df_in.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y = df_in['Participant']
    
    # Guardamos nombres originales para procesar
    original_feature_names = X.columns
    
    # 2. Entrenar Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    
    # 3. Crear DF para visualización (Con Traducción)
    df_imp = pd.DataFrame({
        'Metrica_Original': original_feature_names,
        'Metrica_Esp': [traducir(m) for m in original_feature_names], # Columna traducida
        'Importancia': importances
    }).sort_values(by='Importancia', ascending=False)
    
    # Guardar CSV con el ranking
    df_imp.to_csv(os.path.join(OUTPUT_DIR, "Ranking_Importancia_RF.csv"), index=False)
    
    # 4. Graficar Top 15 (Usando nombres en Español)
    plt.figure(figsize=(12, 8)) # Un poco más ancho para nombres largos
    sns.barplot(data=df_imp.head(15), x='Importancia', y='Metrica_Esp', palette='magma')
    
    plt.title('Top 15 Métricas que distinguen a los usuarios\n(Importancia de Gini - Random Forest)', fontsize=14, weight='bold')
    plt.xlabel('Importancia Relativa')
    plt.ylabel('Métrica Biométrica')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Ranking_Feature_Importance.png"), dpi=300)
    plt.close()
    
    # Retornamos los nombres ORIGINALES (Inglés) para que el código pueda filtrar el DF
    top_metrics_english = df_imp['Metrica_Original'].head(12).tolist()
    
    print("✅ Ranking generado. Top 3 métricas:")
    for i, row in df_imp.head(3).iterrows():
        print(f"   {i+1}. {row['Metrica_Esp']} ({row['Importancia']:.4f})")
    
    return top_metrics_english

# =============================================================================
#  3A. MAPAS 2D (LDA)
# =============================================================================
def generar_mapa_lda_2d(df_in, filename_suffix, title_suffix):
    X = df_in.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y = df_in['Participant']
    participants = y.unique()
    n_classes = len(participants)
    
    if n_classes < 2: return

    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X, y)
    var = lda.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    sns.scatterplot(x=X_lda[:,0], y=X_lda[:,1], hue=y, style=y, 
                    palette='tab10', s=100, alpha=0.8, edgecolor='white', ax=ax)
    
    for idx, p in enumerate(participants):
        mask = (y == p)
        if np.sum(mask) > 5:
            x_p = X_lda[mask, 0]
            y_p = X_lda[mask, 1]
            confidence_ellipse(x_p, y_p, ax, n_std=2.0, edgecolor=colors[idx % len(colors)], linestyle='--', linewidth=2)
            ax.text(np.mean(x_p), np.mean(y_p), p, fontsize=10, weight='bold', 
                    color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title(f'Clusterización LDA 2D - {title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel(f'Discriminante Lineal 1 ({var[0]*100:.1f}%)') # Traducido
    plt.ylabel(f'Discriminante Lineal 2 ({var[1]*100:.1f}%)') # Traducido
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Mapa_2D_{filename_suffix}.png"), dpi=300)
    plt.close()

# =============================================================================
#  3B. MAPAS 3D INTERACTIVOS (LDA) - LEYENDA CORREGIDA
# =============================================================================
def generar_mapa_lda_3d_interactivo(df_in, filename_suffix, title_suffix):
    X = df_in.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y = df_in['Participant']
    participants = y.unique()
    n_classes = len(participants)

    if n_classes < 3: 
        return

    print(f"🧊 Generando 3D: {title_suffix}...")
    
    lda = LDA(n_components=min(3, n_classes - 1))
    X_lda = lda.fit_transform(X, y)
    var = lda.explained_variance_ratio_

    # Si hay menos de 3 componentes (ej. solo 2 clases), rellenamos con ceros para plotear 3D
    if X_lda.shape[1] < 3:
        z_vals = np.zeros(X_lda.shape[0])
    else:
        z_vals = X_lda[:, 2]

    fig = plt.figure(figsize=(14, 10)) # Lienzo más ancho para acomodar la leyenda
    ax = fig.add_subplot(111, projection='3d')
    
    label_encoder = {name: i for i, name in enumerate(participants)}
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for p in participants:
        mask = (y == p)
        idx = label_encoder[p]
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], z_vals[mask], 
                   label=p, s=50, alpha=0.8, color=colors[idx % len(colors)], edgecolors='w')
        
        # Etiqueta en el centroide del cluster
        center = np.mean(X_lda[mask], axis=0)
        z_center = np.mean(z_vals[mask])
        ax.text(center[0], center[1], z_center, p, fontsize=9, weight='bold', color='black')

    ax.set_xlabel(f'LD1 ({var[0]*100:.1f}%)')
    ax.set_ylabel(f'LD2 ({var[1]*100:.1f}%)')
    if len(var) > 2:
        ax.set_zlabel(f'LD3 ({var[2]*100:.1f}%)')
    
    ax.set_title(f'Espacio Biométrico 3D - {title_suffix}', fontsize=14)
    
    # --- CORRECCIÓN DE LA LEYENDA ---
    # bbox_to_anchor=(1.3, 0.9) mueve la caja más a la derecha (eje X > 1.0)
    ax.legend(bbox_to_anchor=(1.3, 0.9), loc='center right', borderaxespad=0.)

    plt.savefig(os.path.join(OUTPUT_DIR, f"Captura_3D_{filename_suffix}.png"), dpi=300, bbox_inches='tight')
    plt.show() 

def ejecutar_lda_escalonado(df):
    print("\n" + "-"*30 + " MAPAS LDA (2D y 3D) " + "-"*30)
    participants = df['Participant'].unique()
    
    if len(participants) >= 5:
        df_5 = df[df['Participant'].isin(participants[:5])]
        generar_mapa_lda_2d(df_5, "5_Personas", "5 Participantes")
        generar_mapa_lda_3d_interactivo(df_5, "5_Personas", "5 Participantes")
        
    generar_mapa_lda_2d(df, "Completo", "Dataset Completo")
    generar_mapa_lda_3d_interactivo(df, "Completo", "Dataset Completo")

# =============================================================================
#  4. CLASIFICACIÓN ESCALONADA (SVM)
# =============================================================================
def generar_matriz_svm(df_in, filename_suffix, title_suffix):
    print(f"🤖 Entrenando SVM: {title_suffix}")
    X = df_in.drop(columns=['Participant', 'VideoID', 'Window_Start'], errors='ignore')
    y = df_in['Participant']
    
    if len(y) < 20: return 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(scaler.fit_transform(X_train), y_train)
    
    y_pred = clf.predict(scaler.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    print(f"   🎯 Accuracy: {acc*100:.2f}%")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title(f'Matriz de Confusión - {title_suffix}\nExactitud: {acc*100:.1f}%', fontsize=14, fontweight='bold')
    plt.ylabel('Clase Real')      # Traducido
    plt.xlabel('Clase Predicha')  # Traducido
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Matriz_Confusion_{filename_suffix}.png"), dpi=300)
    plt.close()
    
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_csv(
        os.path.join(OUTPUT_DIR, f"Reporte_{filename_suffix}.csv"))

def ejecutar_svm_escalonado(df):
    print("\n" + "-"*30 + " CLASIFICACIÓN SVM " + "-"*30)
    participants = df['Participant'].unique()
    
    if len(participants) >= 5:
        generar_matriz_svm(df[df['Participant'].isin(participants[:5])], "5_Personas", "5 Participantes")
    
    generar_matriz_svm(df, "Completo", "Dataset Completo")

# =============================================================================
#  5. RADARES (CON TRADUCCIÓN)
# =============================================================================
def generar_fichas_radar(df, top_metrics_english):
    print("\n" + "-"*30 + " GENERANDO PERFILES DE USUARIO " + "-"*30)
    
    # Filtramos solo las que existen
    metrics_radar = [m for m in top_metrics_english if m in df.columns]
    
    if len(metrics_radar) < 3: 
        print("⚠️ No hay suficientes métricas para el radar.")
        return

    # Creamos las etiquetas traducidas para el gráfico
    labels_espanol = [traducir(m) for m in metrics_radar]

    participants = df['Participant'].unique()
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[metrics_radar] = scaler.fit_transform(df[metrics_radar])
    global_mean = df_scaled[metrics_radar].mean()
    
    N = len(metrics_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
    
    for idx, p in enumerate(participants):
        p_data = df_scaled[df_scaled['Participant'] == p][metrics_radar].mean()
        values = p_data.values.flatten().tolist()
        values += values[:1]
        g_values = global_mean.values.flatten().tolist()
        g_values += g_values[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Fondo (Promedio Global)
        ax.plot(angles, g_values, linewidth=1, linestyle='--', color='gray', alpha=0.5, label='Promedio Global')
        ax.fill(angles, g_values, color='gray', alpha=0.1)
        
        # Perfil del Usuario
        color_line = colors[idx % len(colors)]
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', color=color_line, label=p)
        ax.fill(angles, values, color=color_line, alpha=0.35)
        
        # Puntos destacados
        ax.scatter(angles, values, s=50, c=color_line, edgecolors='white', zorder=10)
        
        # --- ETIQUETAS TRADUCIDAS ---
        # Usamos labels_espanol en lugar de metrics_radar (inglés)
        plt.xticks(angles[:-1], labels_espanol, size=9, weight='bold', color='#333333')
        
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=8)
        plt.ylim(0, 1)
        
        plt.title(f"Perfil Biométrico: {p}", size=18, weight='bold', color=color_line, y=1.08)
        
        # Leyenda desplazada para no tapar
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Perfil_{p}.png"), dpi=300)
        plt.close()
        
    print(f"✅ Fichas generadas con las {len(metrics_radar)} métricas principales.")

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    df = cargar_dataset()
    if df is not None:
        
        # 1. Feature Importance (Obtenemos métricas en Inglés, pero gráfica sale en Español)
        best_metrics_eng = analizar_importancia_features(df)
        
        # 2. Mapas LDA
        ejecutar_lda_escalonado(df) 
        
        # 3. Clasificación SVM
        ejecutar_svm_escalonado(df)
        
        # 4. Radares (Le pasamos las métricas en inglés para filtrar datos, las traduce al pintar)
        generar_fichas_radar(df, best_metrics_eng)
        
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"📂 Archivos en: {OUTPUT_DIR}")
        print("="*70)