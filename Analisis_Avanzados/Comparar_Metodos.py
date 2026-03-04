"""
=============================================================================
ANÁLISIS COMPARATIVO DE TÉCNICAS DE REDUCCIÓN Y CLASIFICACIÓN BIOMÉTRICA
=============================================================================
Técnicas evaluadas:
  Reducción dimensional : PCA · t-SNE · UMAP (si disponible)
  Clustering            : KMeans · DBSCAN
  Clasificación         : Random Forest (hiperparámetros optimizados)

Salida: figuras PNG + CSV de resumen para incluir en la tesis.
=============================================================================
"""

import os, glob, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, davies_bouldin_score
)

# --- Aumentar tamaño de fuente global para mejor legibilidad ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# UMAP opcional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP no disponible — se omite (instala umap-learn para activarlo).")

warnings.filterwarnings('ignore')

# =============================================================================
#  CONFIGURACIÓN
# =============================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
INPUT_PATH  = os.path.join(BASE_DIR, "Analizar_Data", "Resultados", "**", "*_BIOMETRIC_METRICS.csv")

timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, "Analisis_Avanzados", "Comparativa_Tecnicas", f"Comparativa_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hiperparámetros RF optimizados
RF_PARAMS = dict(
    n_estimators     = 300,
    min_samples_split= 2,
    min_samples_leaf = 1,
    max_features     = 'sqrt',
    max_depth        = 50,
    criterion        = 'gini',
    bootstrap        = False,
    random_state     = 42,
    n_jobs           = 1      # n_jobs=-1 causa warnings de joblib en Windows
)

# Colores consistentes por técnica para todas las figuras
COLORES_TECNICA = {
    'PCA'          : '#4C72B0',
    't-SNE'        : '#DD8452',
    'UMAP'         : '#55A868',
    'KMeans'       : '#C44E52',
    'DBSCAN'       : '#8172B2',
    'Random Forest': '#2ca02c',
}

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
#  1. CARGA DE DATOS (con exclusión de "test" y anonimización)
# =============================================================================
def cargar_dataset():
    print("=" * 70)
    print("  ANÁLISIS COMPARATIVO DE TÉCNICAS BIOMÉTRICAS")
    print("=" * 70)
    files = glob.glob(INPUT_PATH, recursive=True)
    if not files:
        print("❌ No se encontraron archivos CSV en:", INPUT_PATH)
        return None

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  ⚠️  Error al leer {f}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    if 'Pupil_Mean' in df.columns:
        df = df[df['Pupil_Mean'] > 0]

    # --- EXCLUIR PARTICIPANTES CON "test" EN SU NOMBRE ---
    df = df[~df['Participant'].str.contains('test', case=False, na=False)]
    
    # --- ANONIMIZAR: renombrar a p1, p2, p3, ... ---
    participantes_unicos = sorted(df['Participant'].unique())
    mapeo = {orig: f'p{i+1}' for i, orig in enumerate(participantes_unicos)}
    df['Participant'] = df['Participant'].map(mapeo)
    
    print(f"✅ Dataset: {len(df)} muestras | {df['Participant'].nunique()} participantes (anonimizados)\n")
    return df


def preparar_XY(df):
    """Devuelve X escalado, y etiquetas, scaler, nombres de features."""
    drop_cols = ['Participant', 'VideoID', 'Window_Start']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y     = df['Participant'].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)
    return X_sc, y, scaler, X_raw.columns.tolist()


# =============================================================================
#  2. REDUCCIÓN DIMENSIONAL: PCA
# =============================================================================
def analizar_pca(X, y, participantes):
    print("─" * 60)
    print("  PCA — Análisis de Componentes Principales")
    print("─" * 60)

    pca_full = PCA()
    pca_full.fit(X)
    var_acum = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = int(np.searchsorted(var_acum, 0.95)) + 1

    print(f"  Varianza acumulada al 95%: {n_95} componentes")
    print(f"  PC1 explica: {pca_full.explained_variance_ratio_[0]*100:.2f}%")
    print(f"  PC2 explica: {pca_full.explained_variance_ratio_[1]*100:.2f}%")
    print(f"  PC3 explica: {pca_full.explained_variance_ratio_[2]*100:.2f}%")

    pca2 = PCA(n_components=2)
    X_2d = pca2.fit_transform(X)

    pca3 = PCA(n_components=3)
    X_3d = pca3.fit_transform(X)

    colores = plt.cm.tab20(np.linspace(0, 1, len(participantes)))
    mapa_color = {p: colores[i] for i, p in enumerate(participantes)}

    # — Figura 1: Scree plot + PCA 2D (combinada) —
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('PCA — Análisis de Componentes Principales', fontsize=18, fontweight='bold')

    # Scree plot
    ax = axes[0]
    ax.bar(range(1, min(16, len(var_acum)+1)),
           pca_full.explained_variance_ratio_[:15] * 100,
           color=COLORES_TECNICA['PCA'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2 = ax.twinx()
    ax2.plot(range(1, min(16, len(var_acum)+1)),
             var_acum[:15] * 100, 'o-', color='red', linewidth=2, markersize=5)
    ax2.axhline(95, color='red', linestyle='--', linewidth=1, alpha=0.6, label='95%')
    ax2.set_ylabel('Varianza acumulada (%)', color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
    ax.set_xlabel('Componente Principal', fontsize=14)
    ax.set_ylabel('Varianza explicada (%)', fontsize=14)
    ax.set_title(f'Scree Plot\n({n_95} componentes para 95% de varianza)', fontsize=14)
    ax2.legend(loc='center right', fontsize=12)

    # PCA 2D scatter
    ax = axes[1]
    for p in participantes:
        mask = y == p
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=mapa_color[p], label=p, s=40, alpha=0.7, edgecolors='w', linewidths=0.5)
    ax.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
    ax.set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
    ax.set_title('Proyección PCA 2D\n(Separabilidad entre clases)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'PCA_Scree_2D.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # — Figura 2: PCA 3D (opcional, comentar si no se necesita) —
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    for p in participantes:
        mask = y == p
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                   color=mapa_color[p], label=p, s=40, alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca_full.explained_variance_ratio_[2]*100:.1f}%)', fontsize=12)
    ax.set_title('Proyección PCA 3D', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'PCA_3D.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Figuras guardadas.\n")
    return X_2d, X_3d, pca_full.explained_variance_ratio_


# =============================================================================
#  3. REDUCCIÓN DIMENSIONAL: t-SNE (simplificado a una sola perplejidad)
# =============================================================================
def analizar_tsne(X, y, participantes):
    print("─" * 60)
    print("  t-SNE — Stochastic Neighbor Embedding")
    print("─" * 60)

    # Reducir con PCA antes de t-SNE (recomendado para alta dimensionalidad)
    n_pre = min(50, X.shape[1])
    X_pre = PCA(n_components=n_pre).fit_transform(X)

    # Usamos perplejidad = 30 como valor representativo
    perp = 30
    print(f"  Calculando t-SNE (perplexity={perp})...")
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000,
                random_state=42, learning_rate='auto', init='pca')
    X_tsne = tsne.fit_transform(X_pre)

    colores = plt.cm.tab20(np.linspace(0, 1, len(participantes)))
    mapa_color = {p: colores[i] for i, p in enumerate(participantes)}

    fig, ax = plt.subplots(figsize=(12, 10))
    for p in participantes:
        mask = y == p
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   color=mapa_color[p], label=p, s=40, alpha=0.7,
                   edgecolors='w', linewidths=0.5)
    ax.set_title(f't-SNE (perplejidad = {perp})', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=14)
    ax.set_ylabel('t-SNE 2', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tSNE.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Figura guardada.\n")
    return X_tsne


# =============================================================================
#  4. REDUCCIÓN DIMENSIONAL: UMAP (simplificado a una configuración)
# =============================================================================
def analizar_umap(X, y, participantes):
    if not UMAP_AVAILABLE:
        print("  ⚠️  UMAP no disponible — omitido.\n")
        return None

    print("─" * 60)
    print("  UMAP — Uniform Manifold Approximation")
    print("─" * 60)

    colores = plt.cm.tab20(np.linspace(0, 1, len(participantes)))
    mapa_color = {p: colores[i] for i, p in enumerate(participantes)}

    # Configuración representativa: n_neighbors=15, min_dist=0.1
    n_neighbors, min_dist = 15, 0.1
    print(f"  Calculando UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42)
    X_u = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 10))
    for p in participantes:
        mask = y == p
        ax.scatter(X_u[mask, 0], X_u[mask, 1],
                   color=mapa_color[p], label=p, s=40, alpha=0.7,
                   edgecolors='w', linewidths=0.5)
    ax.set_title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'UMAP.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Figura guardada.\n")
    return X_u


# =============================================================================
#  5. CLUSTERING: KMeans y DBSCAN (figura compuesta mejorada)
# =============================================================================
def analizar_clustering(X, y, X_2d_pca, participantes):
    print("─" * 60)
    print("  CLUSTERING — KMeans y DBSCAN")
    print("─" * 60)

    n_clusters = len(participantes)
    y_int = pd.factorize(y)[0]   # etiquetas numéricas para métricas

    # ── KMeans ──────────────────────────────────────────────────────────────
    print(f"  Ejecutando KMeans (k={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42, max_iter=500)
    labels_km = km.fit_predict(X)

    ari_km  = adjusted_rand_score(y_int, labels_km)
    nmi_km  = normalized_mutual_info_score(y_int, labels_km)
    sil_km  = silhouette_score(X, labels_km)
    dbi_km  = davies_bouldin_score(X, labels_km)

    print(f"    ARI:              {ari_km:.4f}  (1=perfecto, 0=aleatorio)")
    print(f"    NMI:              {nmi_km:.4f}  (1=perfecto)")
    print(f"    Silhouette:       {sil_km:.4f}  (>0 mejor, -1 peor)")
    print(f"    Davies-Bouldin:   {dbi_km:.4f}  (0 mejor)")

    # — Elbow method —
    inercias = []
    ks = range(2, min(n_clusters + 6, 21))
    for k in ks:
        inercias.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_)

    # ── DBSCAN ──────────────────────────────────────────────────────────────
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5).fit(X)
    dists, _ = nn.kneighbors(X)
    min_samples_auto = max(3, len(y) // (n_clusters * 4))

    best_labels_db, best_eps, best_n_db = None, None, 0
    for percentil in [30, 40, 50, 60, 70, 80, 90]:
        eps_cand = float(np.percentile(dists[:, -1], percentil))
        lbl_cand = DBSCAN(eps=eps_cand, min_samples=min_samples_auto).fit_predict(X)
        n_cand   = len(set(lbl_cand)) - (1 if -1 in lbl_cand else 0)
        if n_cand >= 2:
            best_labels_db, best_eps, best_n_db = lbl_cand, eps_cand, n_cand
            break

    if best_labels_db is None:
        best_eps       = float(np.percentile(dists[:, -1], 50))
        best_labels_db = DBSCAN(eps=best_eps, min_samples=min_samples_auto).fit_predict(X)
        best_n_db      = len(set(best_labels_db)) - (1 if -1 in best_labels_db else 0)

    labels_db, eps_auto, n_clusters_db = best_labels_db, best_eps, best_n_db
    n_ruido_db = int(np.sum(labels_db == -1))
    pct_ruido  = n_ruido_db / len(labels_db) * 100

    print(f"\n  Ejecutando DBSCAN (eps={eps_auto:.3f}, min_samples={min_samples_auto})...")
    print(f"    Clústeres encontrados: {n_clusters_db}  (esperados: {n_clusters})")
    print(f"    Puntos de ruido (-1):  {n_ruido_db} ({pct_ruido:.1f}%)")

    if n_clusters_db >= 2:
        mask_valid = labels_db != -1
        ari_db = adjusted_rand_score(y_int[mask_valid], labels_db[mask_valid])
        nmi_db = normalized_mutual_info_score(y_int[mask_valid], labels_db[mask_valid])
        sil_db = silhouette_score(X[mask_valid], labels_db[mask_valid]) if mask_valid.sum() > 1 else 0.0
        dbi_db = davies_bouldin_score(X[mask_valid], labels_db[mask_valid]) if mask_valid.sum() > 1 else 0.0
    else:
        ari_db = nmi_db = sil_db = dbi_db = 0.0
        print("  ⚠️  DBSCAN no encontró estructura válida — ARI/NMI = 0 "
              "(confirma que clustering no supervisado es insuficiente).")

    print(f"    ARI: {ari_db:.4f}")
    print(f"    NMI: {nmi_db:.4f}")

    # ── FIGURA COMPUESTA (tamaño aumentado) ─────────────────────────────────
    fig = plt.figure(figsize=(24, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Análisis de Clustering sobre Espacio Biométrico',
                 fontsize=20, fontweight='bold')

    # (A) Clases reales en PCA 2D
    ax = fig.add_subplot(gs[0, 0])
    colores_real = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    mapa_real = {p: colores_real[i] for i, p in enumerate(participantes)}
    for p in participantes:
        mask = y == p
        ax.scatter(X_2d_pca[mask, 0], X_2d_pca[mask, 1],
                   color=mapa_real[p], s=40, alpha=0.7, label=p, edgecolors='w', linewidths=0.3)
    ax.set_title('Clases Reales (PCA 2D)', fontsize=16, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)

    # (B) KMeans en PCA 2D
    ax = fig.add_subplot(gs[0, 1])
    pal_km = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    for k in range(n_clusters):
        mask = labels_km == k
        ax.scatter(X_2d_pca[mask, 0], X_2d_pca[mask, 1],
                   color=pal_km[k % len(pal_km)], s=40, alpha=0.6, edgecolors='w', linewidths=0.3)
    pca2_fitted = PCA(n_components=2).fit(X)
    centroids_2d = pca2_fitted.transform(km.cluster_centers_)
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
               s=200, c='black', marker='X', zorder=5, label='Centroides')
    ax.set_title(f'KMeans (k={n_clusters})\nARI={ari_km:.3f}  NMI={nmi_km:.3f}',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)

    # (C) DBSCAN en PCA 2D
    ax = fig.add_subplot(gs[0, 2])
    unique_db = sorted(set(labels_db))
    pal_db = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_db))))
    for i, lbl in enumerate(unique_db):
        mask = labels_db == lbl
        color = 'gray' if lbl == -1 else pal_db[i % len(pal_db)]
        lname = 'Ruido' if lbl == -1 else f'Cluster {lbl}'
        ax.scatter(X_2d_pca[mask, 0], X_2d_pca[mask, 1],
                   color=color, s=40, alpha=0.5, edgecolors='w', linewidths=0.3, label=lname)
    title_db = (f'DBSCAN (eps={eps_auto:.2f})\n'
                f'{n_clusters_db} clústeres | {pct_ruido:.1f}% ruido')
    if not np.isnan(ari_db):
        title_db += f'\nARI={ari_db:.3f}  NMI={nmi_db:.3f}'
    ax.set_title(title_db, fontsize=16, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)

    # (D) Elbow method
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(list(ks), inercias, 'o-', color=COLORES_TECNICA['KMeans'],
            linewidth=2, markersize=8)
    ax.axvline(n_clusters, color='red', linestyle='--', linewidth=2,
               label=f'k={n_clusters} (usado)')
    ax.set_xlabel('Número de clústeres (k)', fontsize=14)
    ax.set_ylabel('Inercia (WCSS)', fontsize=14)
    ax.set_title('Método del Codo — KMeans\n(Justificación de k óptimo)', fontsize=16)
    ax.legend(fontsize=12)

    # (E) Comparativa métricas clustering
    ax = fig.add_subplot(gs[1, 1:])
    metricas_nombres = ['ARI', 'NMI', 'Silhouette']
    vals_km = [ari_km, nmi_km, sil_km]
    vals_db = [ari_db, nmi_db, sil_db if not np.isnan(sil_db) else 0.0]
    x_pos = np.arange(len(metricas_nombres))
    w = 0.35
    bars1 = ax.bar(x_pos - w/2, vals_km, w, label='KMeans',
                   color=COLORES_TECNICA['KMeans'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos + w/2, vals_db, w, label='DBSCAN',
                   color=COLORES_TECNICA['DBSCAN'], edgecolor='black', linewidth=0.5)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metricas_nombres, fontsize=14)
    ax.set_ylabel('Valor de la métrica', fontsize=14)
    ax.set_title('Comparativa KMeans vs DBSCAN\n(Métricas de calidad de clustering)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Clustering_KMeans_DBSCAN.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✅ Figura de clustering guardada.\n")

    return {
        'KMeans': {'ARI': ari_km, 'NMI': nmi_km, 'Silhouette': sil_km,
                   'DBI': dbi_km, 'n_clusters': n_clusters},
        'DBSCAN': {'ARI': ari_db, 'NMI': nmi_db, 'Silhouette': sil_db,
                   'DBI': dbi_db, 'n_clusters': n_clusters_db,
                   'pct_ruido': pct_ruido},
    }


# =============================================================================
#  6. RANDOM FOREST — CLASIFICACIÓN SUPERVISADA
# =============================================================================
def analizar_random_forest(X, y, feature_names, participantes):
    print("─" * 60)
    print("  RANDOM FOREST — Clasificación supervisada")
    print("─" * 60)
    print(f"  Hiperparámetros: {RF_PARAMS}\n")

    # Validación cruzada estratificada (5-fold)
    rf = RandomForestClassifier(**RF_PARAMS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_cv = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"  Validación cruzada (5-fold):")
    print(f"    Accuracy por fold: {[f'{s*100:.2f}%' for s in scores_cv]}")
    print(f"    Media: {scores_cv.mean()*100:.2f}%  ±  {scores_cv.std()*100:.2f}%")

    # Train/Test split para matriz de confusión y reporte
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    acc   = accuracy_score(y_te, y_pred)
    rep   = classification_report(y_te, y_pred, output_dict=True)
    f1_m  = rep['macro avg']['f1-score']
    prec  = rep['macro avg']['precision']
    rec   = rep['macro avg']['recall']

    print(f"\n  Conjunto de test (70/30):")
    print(f"    Accuracy:         {acc*100:.2f}%")
    print(f"    F1-score (macro): {f1_m:.4f}")
    print(f"    Precision macro:  {prec:.4f}")
    print(f"    Recall macro:     {rec:.4f}")

    # — Matriz de confusión normalizada —
    labels_ord = sorted(participantes)
    cm = confusion_matrix(y_te, y_pred, labels=labels_ord, normalize='true')
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels_ord, yticklabels=labels_ord,
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                annot_kws={'size': 12})
    ax.set_title(f'Random Forest — Matriz de Confusión Normalizada\n'
                 f'Accuracy: {acc*100:.2f}%  |  F1: {f1_m:.4f}  |  '
                 f'CV: {scores_cv.mean()*100:.2f}% ± {scores_cv.std()*100:.2f}%',
                 fontsize=16, fontweight='bold')
    ax.set_ylabel('Clase Real', fontsize=14)
    ax.set_xlabel('Clase Predicha', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'RF_Matriz_Confusion.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # — Feature importance —
    importancias = pd.Series(rf.feature_importances_, index=feature_names)
    importancias = importancias.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(14, 8))
    importancias.plot(kind='barh', ax=ax, color=COLORES_TECNICA['Random Forest'],
                      edgecolor='black', linewidth=0.5)
    ax.invert_yaxis()
    for i, (idx, val) in enumerate(importancias.items()):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=11)
    ax.set_title('Random Forest — Importancia de Características (Gini)\nTop 15',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Importancia relativa', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'RF_Feature_Importance.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # — Distribución de scores CV (opcional, comentar si no se necesita) —
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, 6), scores_cv * 100,
           color=COLORES_TECNICA['Random Forest'], edgecolor='black', linewidth=0.5)
    ax.axhline(scores_cv.mean() * 100, color='red', linestyle='--',
               linewidth=2, label=f'Media: {scores_cv.mean()*100:.2f}%')
    ax.fill_between(range(0, 7),
                    (scores_cv.mean() - scores_cv.std()) * 100,
                    (scores_cv.mean() + scores_cv.std()) * 100,
                    alpha=0.15, color='red', label=f'±1σ: {scores_cv.std()*100:.2f}%')
    ax.set_xlabel('Fold de validación cruzada', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xticks(range(1, 6))
    ax.set_title('Random Forest — Estabilidad por Validación Cruzada (5-fold)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim(50, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'RF_CV_Scores.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    print("  ✅ Figuras de Random Forest guardadas.\n")

    return {
        'Accuracy_test': acc,
        'F1_macro':      f1_m,
        'Precision':     prec,
        'Recall':        rec,
        'CV_mean':       scores_cv.mean(),
        'CV_std':        scores_cv.std(),
        'scores_cv':     scores_cv,
        'rep':           rep,
    }


# =============================================================================
#  7. FIGURA MAESTRA COMPARATIVA (mejorada)
# =============================================================================
def figura_comparativa_final(metricas_cluster, metricas_rf):
    print("─" * 60)
    print("  GENERANDO FIGURA COMPARATIVA FINAL...")
    print("─" * 60)

    tecnicas = ['KMeans\n(No superv.)', 'DBSCAN\n(No superv.)', 'Random Forest\n(Superv.)']
    ari_km  = metricas_cluster['KMeans']['ARI']
    ari_db  = metricas_cluster['DBSCAN']['ARI']
    nmi_km  = metricas_cluster['KMeans']['NMI']
    nmi_db  = metricas_cluster['DBSCAN']['NMI']
    sil_km  = metricas_cluster['KMeans']['Silhouette']
    sil_db  = metricas_cluster['DBSCAN']['Silhouette']
    rf_acc  = metricas_rf['Accuracy_test']
    rf_f1   = metricas_rf['F1_macro']

    fig = plt.figure(figsize=(20, 9))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)
    fig.suptitle('Comparativa de Técnicas: Clustering No Supervisado vs '
                 'Clasificación Supervisada (Random Forest)',
                 fontsize=18, fontweight='bold')

    # Panel A: ARI/NMI vs Accuracy/F1
    ax_a = fig.add_subplot(gs[0])
    metricas_labels = ['Calidad de agrupación\n(ARI / Accuracy)', 'Información mutua\n(NMI / F1-macro)']
    vals = [
        [ari_km,  ari_db,  rf_acc],
        [nmi_km,  nmi_db,  rf_f1 ],
    ]
    colores_barras = [COLORES_TECNICA['KMeans'], COLORES_TECNICA['DBSCAN'],
                      COLORES_TECNICA['Random Forest']]
    x = np.arange(len(metricas_labels))
    w = 0.25
    for i, (tec, col) in enumerate(zip(tecnicas, colores_barras)):
        yvals = [vals[m][i] for m in range(len(metricas_labels))]
        bars = ax_a.bar(x + (i - 1) * w, yvals, w, label=tec,
                        color=col, edgecolor='black', linewidth=0.5)
        for bar in bars:
            ax_a.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.01,
                      f'{bar.get_height():.3f}',
                      ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(metricas_labels, fontsize=13)
    ax_a.set_ylabel('Valor de la métrica (0–1)', fontsize=14)
    ax_a.set_ylim(0, 1.2)
    ax_a.legend(fontsize=12)
    ax_a.set_title('Panel A — Calidad de Agrupación / Clasificación',
                   fontsize=15, fontweight='bold')

    # Panel B: resumen tipo "podio"
    ax_b = fig.add_subplot(gs[1])
    podio_labels = ['KMeans', 'DBSCAN', 'Random Forest']
    podio_vals   = [ari_km, metricas_cluster['DBSCAN']['ARI'], rf_acc]
    podio_cols   = [COLORES_TECNICA['KMeans'], COLORES_TECNICA['DBSCAN'],
                    COLORES_TECNICA['Random Forest']]
    bars_pod = ax_b.bar(podio_labels, podio_vals, color=podio_cols,
                        edgecolor='black', linewidth=0.7, width=0.5)
    for bar, val in zip(bars_pod, podio_vals):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f'{val:.3f}\n({val*100:.1f}%)',
                  ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax_b.axhline(0.8, color='green', linestyle='--', linewidth=2,
                 label='Umbral aceptable (0.80)')
    ax_b.set_ylim(0, 1.2)
    ax_b.set_ylabel('Métrica principal (ARI ó Accuracy)', fontsize=14)
    ax_b.set_title('Panel B — Métrica Principal por Técnica\n(ARI para clustering; Accuracy para RF)',
                   fontsize=15, fontweight='bold')
    ax_b.legend(fontsize=12)

    plt.savefig(os.path.join(OUTPUT_DIR, 'COMPARATIVA_FINAL.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✅ Figura comparativa final guardada.\n")


# =============================================================================
#  8. CSV DE RESUMEN PARA TESIS
# =============================================================================
def guardar_resumen(metricas_cluster, metricas_rf):
    filas = []

    km = metricas_cluster['KMeans']
    filas.append({
        'Técnica':       'KMeans',
        'Tipo':          'Clustering (no supervisado)',
        'Métrica_1':     'ARI',
        'Valor_1':       round(km['ARI'], 4),
        'Métrica_2':     'NMI',
        'Valor_2':       round(km['NMI'], 4),
        'Métrica_3':     'Silhouette',
        'Valor_3':       round(km['Silhouette'], 4),
        'Métrica_4':     'Davies-Bouldin',
        'Valor_4':       round(km['DBI'], 4),
        'Observaciones': f"k={km['n_clusters']} clústeres",
    })

    db = metricas_cluster['DBSCAN']
    filas.append({
        'Técnica':       'DBSCAN',
        'Tipo':          'Clustering (no supervisado)',
        'Métrica_1':     'ARI',
        'Valor_1':       round(db['ARI'], 4),
        'Métrica_2':     'NMI',
        'Valor_2':       round(db['NMI'], 4),
        'Métrica_3':     'Silhouette',
        'Valor_3':       round(db['Silhouette'], 4),
        'Métrica_4':     'Davies-Bouldin',
        'Valor_4':       round(db['DBI'], 4),
        'Observaciones': f"{db['n_clusters']} clústeres | {db['pct_ruido']:.1f}% ruido",
    })

    filas.append({
        'Técnica':       'Random Forest',
        'Tipo':          'Clasificación (supervisado)',
        'Métrica_1':     'Accuracy (test)',
        'Valor_1':       round(metricas_rf['Accuracy_test'], 4),
        'Métrica_2':     'F1-score macro',
        'Valor_2':       round(metricas_rf['F1_macro'], 4),
        'Métrica_3':     'CV mean (5-fold)',
        'Valor_3':       round(metricas_rf['CV_mean'], 4),
        'Métrica_4':     'CV std',
        'Valor_4':       round(metricas_rf['CV_std'], 4),
        'Observaciones': f"n_estimators=300, max_depth=50, criterion=gini",
    })

    df_res = pd.DataFrame(filas)
    ruta = os.path.join(OUTPUT_DIR, 'Resumen_Comparativo.csv')
    df_res.to_csv(ruta, index=False)
    print(f"  📄 Resumen CSV guardado en: {ruta}")
    return df_res


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    df = cargar_dataset()
    if df is None:
        exit(1)

    X, y, scaler, feature_names = preparar_XY(df)
    participantes = sorted(df['Participant'].unique())

    # 1. PCA (siempre útil)
    X_pca2, X_pca3, var_ratio = analizar_pca(X, y, participantes)

    # 2. t-SNE (simplificado)
    X_tsne = analizar_tsne(X, y, participantes)

    # 3. UMAP (si disponible, simplificado)
    X_umap = analizar_umap(X, y, participantes)

    # 4. Clustering (figura compuesta clave)
    metricas_cluster = analizar_clustering(X, y, X_pca2, participantes)

    # 5. Random Forest (figuras esenciales)
    metricas_rf = analizar_random_forest(X, y, feature_names, participantes)

    # 6. Figura comparativa final (resumen)
    figura_comparativa_final(metricas_cluster, metricas_rf)

    # 7. Resumen CSV
    df_res = guardar_resumen(metricas_cluster, metricas_rf)

    print("\n" + "=" * 70)
    print("  ANÁLISIS COMPLETADO")
    print(f"  Archivos en: {OUTPUT_DIR}")
    print("=" * 70)
    print("\n  RESUMEN RÁPIDO:")
    print(f"    KMeans ARI:        {metricas_cluster['KMeans']['ARI']:.4f}")
    print(f"    DBSCAN ARI:        {metricas_cluster['DBSCAN']['ARI']:.4f}")
    print(f"    RF Accuracy test:  {metricas_rf['Accuracy_test']*100:.2f}%")
    print(f"    RF CV (5-fold):    {metricas_rf['CV_mean']*100:.2f}% ± {metricas_rf['CV_std']*100:.2f}%")
    print("=" * 70)