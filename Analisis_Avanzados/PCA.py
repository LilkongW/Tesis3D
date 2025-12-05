import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.feature_selection import f_classif, SelectKBest
from matplotlib.patches import Ellipse

# Estilo
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.max_open_warning': 0})

# =============================================================================
#  CONFIGURACIÓN
# =============================================================================
PROJECT_ROOT = r"C:\Users\Victor\Documents\Tesis3D"
INPUT_DIR = os.path.join(PROJECT_ROOT, "Analizar_Data", "Resultados")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Analisis_Avanzados", "Resultados_Por_Varianza")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Umbral de Varianza Explicada Deseada (90%)
TARGET_VARIANCE = 0.95

VARIANCE_THRESHOLD = 1e-5     
CORRELATION_THRESHOLD = 0.95  

def cargar_datos():
    print(f"🔍 Buscando archivos en: {INPUT_DIR} ...")
    files = glob.glob(os.path.join(INPUT_DIR, "**", "*METRICS.csv"), recursive=True)
    
    if not files:
        print("❌ No se encontraron archivos.")
        return None
    
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            subj = os.path.basename(f).split('_')[0]
            df['Sujeto'] = subj
            df_list.append(df)
            print(f"   ✓ {os.path.basename(f)} -> {subj}")
        except: pass
        
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"📊 Total: {len(full_df)} muestras | Sujetos: {full_df['Sujeto'].unique()}")
    return full_df

def preprocesar_inicial(df):
    """Limpieza básica."""
    y = df['Sujeto']
    meta = ['VideoID', 'Sujeto', 'Archivo', 'Blink_Rate_Hz', 'Blink_Interval_CV']
    X = df.drop(columns=[c for c in meta if c in df.columns], errors='ignore')
    
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    X.fillna(0, inplace=True)
    
    # Eliminar constantes y redundantes
    X = X.loc[:, X.var() > VARIANCE_THRESHOLD]
    
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
    X = X.drop(columns=to_drop)
    
    print(f"\n🧹 Features iniciales (limpias): {X.shape[1]}")
    return X, y

def seleccionar_por_varianza(X, y):
    """
    Determina K métricas basándose en cuántos componentes explican el 90% de varianza.
    """
    print(f"\n🚀 SELECCIÓN AUTOMÁTICA (CRITERIO: {TARGET_VARIANCE*100}% VARIANZA)")
    
    # 1. PCA Preliminar para determinar dimensionalidad
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca_test = PCA()
    pca_test.fit(X_scaled)
    
    cumsum = np.cumsum(pca_test.explained_variance_ratio_)
    
    # Encontrar K (número de componentes para llegar al 90%)
    k_components = np.argmax(cumsum >= TARGET_VARIANCE) + 1
    current_var = cumsum[k_components-1]
    
    print("-" * 50)
    print(f"📊 ANÁLISIS DE DIMENSIONALIDAD:")
    print(f"   • Varianza explicada por 1 componente:  {cumsum[0]:.2%}")
    print(f"   • Varianza explicada por 2 componentes: {cumsum[1]:.2%}")
    if len(cumsum) > 2:
        print(f"   • Varianza explicada por 3 componentes: {cumsum[2]:.2%}")
    print("-" * 50)
    print(f"💡 CONCLUSIÓN MATEMÁTICA:")
    print(f"   Se necesitan {k_components} dimensiones para explicar el {current_var:.1%} de la varianza.")
    print("-" * 50)
    
    # 2. Seleccionar las Top K métricas usando ANOVA
    # Usamos ANOVA porque queremos las que mejor SEPARAN a los sujetos, 
    # pero limitamos la cantidad a lo que el PCA nos sugirió (K).
    X_df = pd.DataFrame(X_scaled, columns=X.columns)
    selector = SelectKBest(f_classif, k=k_components)
    selector.fit(X_df, y)
    
    # Obtener nombres
    cols = X.columns[selector.get_support()]
    scores = selector.scores_[selector.get_support()]
    
    # Ordenar para mostrar
    selection_df = pd.DataFrame({'Metrica': cols, 'Score': scores}).sort_values('Score', ascending=False)
    
    print(f"\n✅ SELECCIONANDO LAS TOP {k_components} MÉTRICAS:")
    print(selection_df.to_string(index=False))
    
    return X[selection_df['Metrica'].tolist()], selection_df['Metrica'].tolist()

def ejecutar_analisis_final(X_opt, y):
    """PCA y Clustering con el subset seleccionado."""
    print("\n⚙️ Ejecutando Análisis Final...")
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_opt)
    n_features = X_scaled.shape[1]
    
    # PCA para visualización
    if n_features >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        var = pca.explained_variance_ratio_
        xlabel = f"PC1 ({var[0]:.1%})"
        ylabel = f"PC2 ({var[1]:.1%})"
    else:
        # Caso raro de 1 sola dimensión
        X_pca = np.zeros((X_scaled.shape[0], 2))
        X_pca[:, 0] = X_scaled.flatten()
        xlabel = X_opt.columns[0]
        ylabel = "N/A"
        var = [1.0, 0.0]

    # Clustering K-Means
    km = KMeans(n_clusters=2, random_state=42, n_init=50)
    y_pred = km.fit_predict(X_scaled)
    
    # ARI Final
    ari_final = adjusted_rand_score(pd.factorize(y)[0], y_pred)
    
    # Visualización
    plt.figure(figsize=(10, 8))
    subjects = y.unique()
    colors = sns.color_palette("bright", len(subjects))
    color_map = dict(zip(subjects, colors))
    
    for subj in subjects:
        idx = y == subj
        # Jitter para visualización 1D si fuera necesario
        y_vals = X_pca[idx, 1] if n_features >= 2 else np.random.normal(0, 0.02, size=sum(idx))
        
        plt.scatter(X_pca[idx, 0], y_vals, c=[color_map[subj]], label=f"{subj}", s=120, edgecolors='k', alpha=0.8)
        
        # Elipses (solo si 2D real)
        pts = np.column_stack((X_pca[idx, 0], y_vals))
        if len(pts) > 2 and n_features >= 2:
            try:
                cov = np.cov(pts, rowvar=False)
                l, v = np.linalg.eig(cov)
                l = np.sqrt(l)
                ell = Ellipse(xy=np.mean(pts, axis=0), width=l[0]*4, height=l[1]*4,
                              angle=np.rad2deg(np.arccos(v[0, 0])),
                              edgecolor=color_map[subj], facecolor='none', linestyle='--', linewidth=2)
                plt.gca().add_artist(ell)
            except: pass

    plt.title(f"Clustering Basado en Varianza ({n_features} métricas)\nARI: {ari_final:.4f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, "Mapa_Varianza.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    # Matriz Confusión
    cm = confusion_matrix(pd.factorize(y)[0], y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f"Matriz Confusión\nARI: {ari_final:.2f}")
    plt.ylabel("Real")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Matriz_Confusion.png"))
    plt.close()

def main():
    df = cargar_datos()
    if df is None: return
    if df['Sujeto'].nunique() < 2:
        print("⚠️ Se necesitan al menos 2 sujetos.")
        return

    # 1. Limpieza
    X, y = preprocesar_inicial(df)
    
    # 2. Selección Inteligente (Varianza -> ANOVA)
    X_opt, feats = seleccionar_por_varianza(X, y)
    
    # 3. Análisis Final
    ejecutar_analisis_final(X_opt, y)
    
    print(f"\n✅ Análisis completado. Resultados en:\n   {OUTPUT_DIR}")

if __name__ == "__main__":
    main()