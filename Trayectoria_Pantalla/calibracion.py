import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
#        CONFIGURACIÓN DE USUARIO
# ==========================================
NAME = "Victor"
# Usamos el Experimento 1 (9 puntos) o 2 (5 puntos) para calibrar
EXP_NUM = 1 

# RUTAS AUTOMÁTICAS
BASE_DIR = r"C:\Users\Victor\Documents\Tesis3D"
INPUT_DATA_DIR = os.path.join(BASE_DIR, "Data", f"Experimento_{EXP_NUM}", f"{NAME}_data")
INPUT_STIM_DIR = os.path.join(BASE_DIR, "Videos", f"Experimento_{EXP_NUM}", NAME)

# Archivo donde se guardará la matriz
OUTPUT_MATRIX_FILE = os.path.join(BASE_DIR, f"calibracion_{NAME}.npy")

# PARÁMETROS
SCREEN_W, SCREEN_H = 1920, 1080
SKIP_INITIAL_MS = 400  # Ignorar los primeros ms tras el cambio de punto (latencia sacádica)

# ==========================================
# 1. FUNCIONES MATEMÁTICAS
# ==========================================
def project_vector_to_plane(vectors_3d):
    """
    Convierte vectores 3D del ojo (x, y, z) en puntos 2D proyectados (x/|z|, y/|z|).
    Esto linealiza la perspectiva cónica del ojo a un plano 2D abstracto.
    """
    z = vectors_3d[:, 2]
    z[z == 0] = 1e-6 
    x_proj = vectors_3d[:, 0] / np.abs(z)
    y_proj = vectors_3d[:, 1] / np.abs(z)
    return np.column_stack((x_proj, y_proj))

def find_homography_manual(src_pts, dst_pts):
    """Calcula la matriz H que mapea src (ojo) -> dst (pantalla)."""
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
    """Aplica la matriz H a puntos 2D."""
    points_hom = np.concatenate([points_2d, np.ones((points_2d.shape[0], 1))], axis=1)
    transformed = np.dot(points_hom, H.T)
    w = transformed[:, 2:]
    w[w == 0] = 1e-10
    return transformed[:, :2] / w

# ==========================================
# 2. LÓGICA DE EXTRACCIÓN SINCRONIZADA
# ==========================================
def extraer_puntos_calibracion(data_path, stim_path):
    print(f"Procesando: {os.path.basename(data_path)}")
    
    # Cargar datos
    df_data = pd.read_csv(data_path)
    df_stim = pd.read_csv(stim_path)
    
    # Filtrar solo validos
    if 'valid_deteccion' in df_data.columns:
        if df_data['valid_deteccion'].dtype == object:
             df_data['valid_deteccion'] = df_data['valid_deteccion'].astype(str).map({'True': True, 'False': False})
        df_data = df_data[df_data['valid_deteccion'] == True].copy()
    
    # Crear eje de tiempo en segundos para la data del ojo
    # Asumimos que timestamp_ms empieza en 0 o está alineado
    df_data['time_s'] = df_data['timestamp_ms'] / 1000.0
    
    # Detectar cambios de posición en el estímulo
    # Agrupamos por coordenadas únicas consecutivas
    df_stim['cambio'] = (df_stim['stimulus_x'] != df_stim['stimulus_x'].shift()) | \
                        (df_stim['stimulus_y'] != df_stim['stimulus_y'].shift())
    df_stim['grupo'] = df_stim['cambio'].cumsum()
    
    src_puntos = [] # Ojo (Proyectado)
    dst_puntos = [] # Pantalla (Normalizado 0-1)
    
    # Iterar sobre cada punto de fijación mostrado
    for _, grupo in df_stim.groupby('grupo'):
        # Coordenadas del target (normalizadas si el log guarda pixeles, convertimos)
        # Asumimos que el log guarda píxeles (ej. 960, 540)
        target_x_px = grupo['stimulus_x'].iloc[0]
        target_y_px = grupo['stimulus_y'].iloc[0]
        
        # Normalizar a 0-1 (necesario para la homografía estable)
        target_norm_x = target_x_px / SCREEN_W
        target_norm_y = target_y_px / SCREEN_H
        
        # Tiempos de inicio y fin de este punto
        t_inicio = grupo['relative_time_s'].iloc[0]
        t_fin = grupo['relative_time_s'].iloc[-1]
        
        # Aplicar Delay (Saltar la sacada inicial)
        t_inicio_valid = t_inicio + (SKIP_INITIAL_MS / 1000.0)
        
        # Extraer datos del ojo en esa ventana temporal
        subset_ojo = df_data[(df_data['time_s'] >= t_inicio_valid) & (df_data['time_s'] <= t_fin)]
        
        if len(subset_ojo) > 10: # Necesitamos al menos unos cuantos frames válidos
            # 1. Obtener vectores raw
            vecs = subset_ojo[['gaze_x', 'gaze_y', 'gaze_z']].values.astype(np.float32)
            
            # 2. Proyectar a plano 2D
            proyectados = project_vector_to_plane(vecs)
            
            # 3. Calcular el promedio ROBUSTO (Mediana para ignorar outliers)
            centroide_ojo = np.median(proyectados, axis=0)
            
            src_puntos.append(centroide_ojo)
            dst_puntos.append([target_norm_x, target_norm_y])
            
            print(f"   -> Punto en ({target_x_px}, {target_y_px}): {len(subset_ojo)} samples válidos.")
        else:
            print(f"   ⚠️ Pocos datos para punto en ({target_x_px}, {target_y_px}). Ignorado.")

    return np.array(src_puntos), np.array(dst_puntos)

# ==========================================
#              MAIN
# ==========================================
def main():
    print(f"--- CALIBRACIÓN DETERMINISTA ({NAME}) ---")
    
    # Buscar el primer archivo de data disponible (o iterar sobre varios si quieres fusionar)
    data_files = glob.glob(os.path.join(INPUT_DATA_DIR, "*_data.csv"))
    
    if not data_files:
        print("❌ No hay archivos de data.")
        return
    
    # Usamos el primer archivo encontrado (Intento 1 suele ser el mejor)
    # Puedes cambiar esto para usar un archivo específico
    data_path = data_files[0] 
    
    # Buscar el log correspondiente
    filename = os.path.basename(data_path)
    stim_filename = filename.replace("_data.csv", "_stimulus.csv")
    stim_path = os.path.join(INPUT_STIM_DIR, stim_filename)
    
    if not os.path.exists(stim_path):
        print(f"❌ ERROR FATAL: No se encontró el log de estímulos: {stim_path}")
        print("   Ejecuta 'experimentos_webcam120fps.py' con logging activado primero.")
        return

    # 1. Extraer pares de puntos (Ojo -> Pantalla)
    src_pts, dst_pts = extraer_puntos_calibracion(data_path, stim_path)
    
    if len(src_pts) < 4:
        print("❌ No hay suficientes puntos válidos para calcular la homografía (Min 4).")
        return

    # 2. Calcular Matriz H
    H_matrix = find_homography_manual(src_pts, dst_pts)
    
    # 3. Guardar
    np.save(OUTPUT_MATRIX_FILE, H_matrix)
    print(f"\n✅ Matriz guardada en: {OUTPUT_MATRIX_FILE}")
    print("   Ahora puedes usar esta matriz para mapear mirada a pantalla.")

    # ==========================================
    # 4. VALIDACIÓN VISUAL
    # ==========================================
    print("\n--- Generando Gráfica de Validación ---")
    
    # Aplicar la matriz a los puntos de origen para ver qué tan bien quedaron
    projected_targets = apply_homography(src_pts, H_matrix)
    
    # Des-normalizar a píxeles para graficar
    real_targets_px = dst_pts * [SCREEN_W, SCREEN_H]
    estimated_px = projected_targets * [SCREEN_W, SCREEN_H]
    
    # Calcular error en píxeles
    errors = np.linalg.norm(real_targets_px - estimated_px, axis=1)
    mean_error = np.mean(errors)
    print(f"Error medio de calibración: {mean_error:.2f} px")
    
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    
    # Dibujar pantalla
    plt.plot([0, SCREEN_W, SCREEN_W, 0, 0], [0, 0, SCREEN_H, SCREEN_H, 0], 'w-', lw=2)
    
    # Dibujar puntos
    plt.scatter(real_targets_px[:, 0], real_targets_px[:, 1], c='white', s=200, label='Target Real', edgecolors='red', zorder=2)
    plt.scatter(estimated_px[:, 0], estimated_px[:, 1], c='cyan', marker='x', s=200, label='Estimación', linewidth=3, zorder=3)
    
    # Dibujar líneas de error
    for i in range(len(real_targets_px)):
        plt.arrow(real_targets_px[i,0], real_targets_px[i,1], 
                  estimated_px[i,0]-real_targets_px[i,0], estimated_px[i,1]-real_targets_px[i,1], 
                  color='yellow', alpha=0.5, zorder=1)

    plt.title(f"Resultado Calibración (Determinista) - Error Medio: {mean_error:.1f} px", fontsize=16)
    plt.xlabel(f"Matriz guardada en: {os.path.basename(OUTPUT_MATRIX_FILE)}")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()