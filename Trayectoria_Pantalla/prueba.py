import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

# ==========================================
# 1. FUNCIONES MATEMÁTICAS (Idénticas al anterior)
# ==========================================
def project_vector_to_plane(vectors_3d):
    z = vectors_3d[:, 2]
    z[z == 0] = 1e-6 
    x_proj = vectors_3d[:, 0] / np.abs(z)
    y_proj = vectors_3d[:, 1] / np.abs(z)
    return np.column_stack((x_proj, y_proj))

def find_homography_manual(src_pts, dst_pts):
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    U, S, Vh = np.linalg.svd(np.array(A))
    return Vh[-1, :].reshape(3, 3)

def apply_homography(points_2d, H):
    points_hom = np.concatenate([points_2d, np.ones((points_2d.shape[0], 1))], axis=1)
    transformed = np.dot(points_hom, H.T)
    w = transformed[:, 2:]
    w[w == 0] = 1e-10
    return transformed[:, :2] / w

def sort_grid_points(points):
    points = points[points[:, 1].argsort()]
    row1 = points[0:3][points[0:3][:, 0].argsort()]
    row2 = points[3:6][points[3:6][:, 0].argsort()]
    row3 = points[6:9][points[6:9][:, 0].argsort()]
    return np.vstack((row1, row2, row3))

# Configuración
SCREEN_W = 1920
SCREEN_H = 1080

# ==========================================
# 2. PROCESAMIENTO
# ==========================================
# --- ENTRENAMIENTO ---
path_train = r'C:\Users\Victor\Documents\Tesis3D\Data\Experimento_1\Victor_data\Victor_experimento_1_intento_1_data.csv'
df_train = pd.read_csv(path_train)
df_train = df_train[df_train['valid_deteccion']]
vec_train = df_train[['gaze_x', 'gaze_y', 'gaze_z']].values.astype(np.float32)
centers_train = sort_grid_points(KMeans(n_clusters=9, random_state=42, n_init='auto').fit(project_vector_to_plane(vec_train)).cluster_centers_)

targets_norm = np.array([[x, y] for y in [0.2, 0.5, 0.8] for x in [0.2, 0.5, 0.8]], dtype=np.float32)
targets_sorted = sort_grid_points(targets_norm)
H_matrix = find_homography_manual(centers_train, targets_sorted)

# --- PRUEBA ---
path_test = r'C:\Users\Victor\Documents\Tesis3D\Data\Experimento_1\Victor_data\Victor_experimento_1_intento_2_data.csv'
df_test = pd.read_csv(path_test)
df_test = df_test[df_test['valid_deteccion']]
vec_test = df_test[['gaze_x', 'gaze_y', 'gaze_z']].values.astype(np.float32)
mapped_test_px = apply_homography(project_vector_to_plane(vec_test), H_matrix) * [SCREEN_W, SCREEN_H]

# Errores
kmeans_test = KMeans(n_clusters=9, random_state=42, n_init='auto').fit(mapped_test_px)
centers_test_px = sort_grid_points(kmeans_test.cluster_centers_)
targets_px = targets_sorted * [SCREEN_W, SCREEN_H]
errors = np.linalg.norm(centers_test_px - targets_px, axis=1)
mean_error = np.mean(errors)

# ==========================================
# 3. VISUALIZACIÓN (LAYOUT EXTERNO)
# ==========================================
# Hacemos la figura más ancha (18 pulgadas) para dejar espacio lateral
fig, ax = plt.subplots(figsize=(18, 9))
plt.style.use('dark_background')
ax.set_facecolor('#0f0f0f')

# Ajuste de Márgenes CLAVE: Dejamos espacio a la derecha (right=0.75)
plt.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.05)

# Pantalla y Datos
ax.add_patch(patches.Rectangle((0, 0), SCREEN_W, SCREEN_H, linewidth=4, edgecolor='#333333', facecolor='none', zorder=0))
ax.scatter(mapped_test_px[:, 0], mapped_test_px[:, 1], c='#00ffff', s=5, alpha=0.15)

for i in range(9):
    tx, ty = targets_px[i]
    cx, cy = centers_test_px[i]
    ax.add_patch(plt.Circle((tx, ty), 40, color='white', zorder=10))
    ax.scatter(cx, cy, c='#ffcc00', marker='+', s=200, linewidth=3, zorder=11)
    ax.arrow(tx, ty, cx-tx, cy-ty, color='#ff5555', width=2, head_width=15, alpha=0.8, zorder=5)
    ax.text(tx, ty, str(i), color='black', fontsize=14, ha='center', va='center', fontweight='bold', zorder=12)

# Título y Ejes
ax.set_title("Validación Cruzada: Victoria 1 vs Victoria 2", fontsize=20, color='white')
ax.set_xlim(-100, SCREEN_W + 100)
ax.set_ylim(SCREEN_H + 100, -100)
ax.set_xlabel('Píxeles X'); ax.set_ylabel('Píxeles Y')

# --- ELEMENTOS EXTERNOS (SIDEBAR) ---

# 1. LEYENDA (Arriba a la derecha, fuera del gráfico)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Target Real', markerfacecolor='white', markersize=10),
    Line2D([0], [0], marker='+', color='w', label='Centro Detectado', markerfacecolor='#ffcc00', markeredgecolor='#ffcc00', markersize=15),
    Line2D([0], [0], marker='o', color='w', label='Muestras (Dispersión)', markerfacecolor='#00ffff', markeredgecolor='none', alpha=0.5, markersize=8),
    Line2D([0], [0], color='#ff5555', lw=2, label='Desviación (Drift)')
]
# bbox_to_anchor=(x, y): (1.05, 1.0) coloca la esquina superior izq de la leyenda justo a la derecha del plot
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.0), 
          frameon=True, facecolor='#222222', edgecolor='gray', title="LEYENDA", title_fontsize=12, fontsize=10)

# 2. CUADRO DE ESTADÍSTICAS (Debajo de la leyenda)
stats_text = (
    f"RESUMEN DE PRECISIÓN\n"
    f"─────────────────────\n"
    f"Error Promedio: {mean_error:6.2f} px\n"
    f"Error Máximo:   {np.max(errors):6.2f} px\n"
    f"Error Mínimo:   {np.min(errors):6.2f} px\n\n"
    f"Resolución:     {SCREEN_W}x{SCREEN_H}"
)
props = dict(boxstyle='round', facecolor='#222222', alpha=1.0, edgecolor='gray')
# Posicionamos verticalmente un poco más abajo (y=0.6)
ax.text(1.05, 0.6, stats_text, transform=ax.transAxes, fontsize=11, color='white', 
        verticalalignment='top', bbox=props, fontfamily='monospace')

plt.show() # O plt.savefig('resultado_final.png')