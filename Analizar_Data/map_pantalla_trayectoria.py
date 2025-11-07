import pandas as pd
import numpy as np
import pygame 
import matplotlib.pyplot as plt
import math
import sys 
import os
from scipy.signal import savgol_filter 
import cv2 # ¡Necesario para la homografía!
from sklearn.cluster import KMeans # ¡NUEVO! Para la calibración robusta

# --- 0. CONFIGURACIÓN DE RUTAS ---
# !!! Modifica estas rutas para que coincidan con tu PC !!!

# Ruta a la CARPETA que contiene los CSV de calibración (Experimento 2, 5 puntos)
CALIBRATION_DATA_PATH = "Data/Experimento_2/Sanchez_data"

# Ruta al ARCHIVO CSV de la espiral que quieres analizar (Experimento 3)
SPIRAL_DATA_PATH = "C:\\Users\\Victor\\Documents\\Tesis3D\\Data\\Experimento_3\\Victor_data\\Victor_intento_1_data.csv"

# --- CONFIGURACIÓN DEL ANÁLISIS ---
FPS = 60.0
SPIRAL_OFFSET_MS = 1500.0 # Offset para el inicio de la espiral
SAVGOL_WINDOW = 7      # Ventana del filtro Sav-Gol (impar)
SAVGOL_POLY = 3        # Orden polinomial del filtro

# ==============================================================================
# FUNCIÓN DE CALIBRACIÓN (¡MODIFICADA CON TU LÓGICA DE K-MEANS!)
# ==============================================================================
def sort_gaze_centers_5_points(centros_gaze):
    """
    Ordena los 5 centros de gaze (encontrados por K-Means) para que coincidan
    con el orden de la animación: [TL, TR, BL, BR, C].
    Asume que Gaze_Y más alto = "arriba" en la pantalla.
    """
    # 1. Ordenar por Y descendente (arriba primero)
    centros_ordenados_y = centros_gaze[np.argsort(-centros_gaze[:, 1])] 
    
    # 2. Separar los 2 de arriba, 1 de centro, 2 de abajo
    top_2_candidatos = centros_ordenados_y[0:2]
    bottom_2_candidatos = centros_ordenados_y[3:5]
    centro_gaze = centros_ordenados_y[2] # El de en medio
    
    # 3. Ordenar los 2 de arriba por X (izquierda-derecha)
    top_2_ordenados = top_2_candidatos[np.argsort(top_2_candidatos[:, 0])] # X ascendente
    
    # 4. Ordenar los 2 de abajo por X
    bottom_2_ordenados = bottom_2_candidatos[np.argsort(bottom_2_candidatos[:, 0])] # X ascendente
    
    # 5. Reconstruir en el orden de la animación
    puntos_gaze_ordenados = [
        top_2_ordenados[0],    # Top-Left
        top_2_ordenados[1],    # Top-Right
        bottom_2_ordenados[0], # Bottom-Left
        bottom_2_ordenados[1], # Bottom-Right
        centro_gaze            # Center
    ]
    return np.array(puntos_gaze_ordenados)

def calculate_calibration_matrix(data_path, WIDTH, HEIGHT):
    """
    Calcula la matriz de homografía usando K-Means en todos los
    archivos de calibración.
    """
    
    print("Iniciando análisis de calibración (Método K-Means)...")
    
    # --- 1. Definir Ground Truth (Píxeles) ---
    cell_width = WIDTH // 3
    cell_height = HEIGHT // 3
    pos_top_left = (cell_width // 2, cell_height // 2) # (320, 180)
    pos_top_right = (2 * cell_width + cell_width // 2, cell_height // 2) # (1600, 180)
    pos_bottom_left = (cell_width // 2, 2 * cell_height + cell_height // 2) # (320, 900)
    pos_bottom_right = (2 * cell_width + cell_width // 2, 2 * cell_height + cell_height // 2) # (1600, 900)
    pos_center = (cell_width + cell_width // 2, cell_height + cell_height // 2) # (960, 540)
    
    # El orden DEBE coincidir con sort_gaze_centers_5_points
    pixel_sequence = [
        pos_top_left, pos_top_right, pos_bottom_left, pos_bottom_right, pos_center
    ]
    
    # --- 2. Encontrar todos los CSV en la carpeta ---
    try:
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and f.startswith('Victor_intento_')]
    except FileNotFoundError:
        print(f"  ERROR: La carpeta de calibración no existe: {data_path}")
        return None
        
    print(f"  Se encontraron {len(csv_files)} archivos CSV de calibración: {csv_files}")

    all_pts_gaze = []
    all_pts_pixel = []

    # --- 3. Procesar cada CSV de calibración con K-Means ---
    for csv_file in csv_files:
        full_path = os.path.join(data_path, csv_file)
        try:
            print(f"  Procesando archivo de calibración: {csv_file}...")
            df = pd.read_csv(full_path)
            
            df_valid = df[df['valid_deteccion'] == True].copy()
            if df_valid.empty:
                print(f"    ADVERTENCIA: No hay datos válidos en {csv_file}. Saltando.")
                continue

            # Extraer todos los datos de mirada válidos
            gaze_data = df_valid[['gaze_x', 'gaze_y']].values
            
            # --- Aquí está tu lógica de K-Means ---
            print("    Buscando 5 centros de fijación (K-Means)...")
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(gaze_data)
            centros_gaze_kmeans = kmeans.cluster_centers_
            
            # --- Aquí está tu lógica de ordenamiento ---
            puntos_gaze_ordenados = sort_gaze_centers_5_points(centros_gaze_kmeans)
            
            # Almacenar los 5 pares de puntos de este archivo
            all_pts_gaze.extend(puntos_gaze_ordenados)
            all_pts_pixel.extend(pixel_sequence)
        
        except Exception as e:
            print(f"  ERROR procesando {csv_file}: {e}")

    # --- 4. Calcular la Matriz de Homografía ---
    if len(all_pts_gaze) < 5:
        print("\nERROR FATAL: No se pudieron extraer suficientes puntos de calibración.")
        return None

    print(f"\n  Cálculo de homografía con {len(all_pts_gaze)} pares de puntos.")
    
    # Convertir listas a arrays de NumPy
    pts_gaze = np.array(all_pts_gaze, dtype=np.float32)
    pts_pixel = np.array(all_pts_pixel, dtype=np.float32)

    # Calcular la matriz de transformación
    M, mask = cv2.findHomography(pts_gaze, pts_pixel, cv2.RANSAC, 5.0)
    
    return M

# ==============================================================================
# SCRIPT DE ANÁLISIS PRINCIPAL
# ==============================================================================
def main_analysis():
    
    # --- PASO 1: OBTENER DIMENSIONES DE PANTALLA ---
    print("Obteniendo resolución de pantalla con Pygame...")
    try:
        pygame.init()
        display_info = pygame.display.Info()
        WIDTH, HEIGHT = display_info.current_w, display_info.current_h
        pygame.quit()
        print(f"Resolución detectada: {WIDTH}x{HEIGHT}")
    except Exception as e:
        print(f"Error al iniciar Pygame: {e}. Usando 1920x1080.")
        WIDTH, HEIGHT = 1920, 1080

    # --- PASO 2: CALCULAR MATRIZ DE CALIBRACIÓN 'M' ---
    print("\n--- PASO 1: CALCULANDO MATRIZ DE CALIBRACIÓN (K-MEANS) ---")
    M = calculate_calibration_matrix(CALIBRATION_DATA_PATH, WIDTH, HEIGHT)
    
    if M is None:
        print("Error fatal: No se pudo calcular la matriz de calibración.")
        sys.exit()
        
    print("--- Matriz de Calibración 'M' obtenida exitosamente ---")
    print(M)

    # --- PASO 3: GENERAR LA "VERDAD ABSOLUTA" (Ground Truth) DE LA ESPIRAL ---
    print("\n--- PASO 2: GENERANDO TRAYECTORIA 'GROUND TRUTH' DE LA ESPIRAL ---")
    CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
    k = 1.0
    theta_max = 15 * np.pi
    velocity = 230 # Velocidad objetivo
    max_r = k * theta_max
    A = (WIDTH / 2) / max_r * 0.95 
    B = (HEIGHT / 2) / max_r * 0.95 
    num_points = 5000
    theta_values = np.linspace(0, theta_max, num_points)
    r_values = k * theta_values
    x_values = CENTER_X + A * r_values * np.cos(theta_values)
    y_values = CENTER_Y + B * r_values * np.sin(theta_values)
    arc_length = np.zeros(len(x_values))
    for i in range(1, len(x_values)):
        dx = x_values[i] - x_values[i - 1]; dy = y_values[i] - y_values[i - 1]
        arc_length[i] = arc_length[i - 1] + np.sqrt(dx**2 + dy**2)
    total_length = arc_length[-1]
    time_values = arc_length / total_length
    total_time = total_length / velocity
    num_frames = int(total_time * FPS)
    interp_time = np.linspace(0, 1, num_frames)
    interp_x = np.interp(interp_time, time_values, x_values)
    interp_y = np.interp(interp_time, time_values, y_values)
    interp_x = interp_x[::-1]
    interp_y = interp_y[::-1]
    ground_truth_path = np.stack((interp_x, interp_y), axis=1)

    # Calcular velocidad 'Ground Truth'
    print("Calculando velocidad 'Ground Truth' (Estímulo)...")
    dt_truth_s = 1.0 / FPS
    dx_truth = np.diff(interp_x)
    dy_truth = np.diff(interp_y)
    distances_truth = np.sqrt(dx_truth**2 + dy_truth**2)
    velocities_truth_px_s = distances_truth / dt_truth_s
    mean_velocity_truth = np.mean(velocities_truth_px_s)
    print(f"Velocidad 'Ground Truth' (Calculada): {mean_velocity_truth:.2f} px/s")

    # --- PASO 4: CARGAR Y FILTRAR DATOS DE LA ESPIRAL ---
    print(f"\n--- PASO 3: CARGANDO Y PROCESANDO DATOS DE LA ESPIRAL ---")
    print(f"Cargando datos de la espiral desde: {SPIRAL_DATA_PATH}")
    try:
        df_gaze = pd.read_csv(SPIRAL_DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo. {e}")
        sys.exit()

    df_valid = df_gaze[df_gaze['valid_deteccion'] == True].copy()
    if df_valid.empty:
        print("No se encontraron detecciones válidas.")
        sys.exit()
        
    df_valid['timestamp_ms'] = df_valid['timestamp_ms'] - SPIRAL_OFFSET_MS
    df_valid = df_valid[df_valid['timestamp_ms'] >= 0]
    if df_valid.empty:
        print("No se encontraron datos válidos después de aplicar el OFFSET.")
        sys.exit()
        
    gaze_raw_x = df_valid['gaze_x'].values
    gaze_raw_y = df_valid['gaze_y'].values

    # Aplicar Filtro Savitzky-Golay
    print(f"Aplicando Filtro Savitzky-Golay (Ventana={SAVGOL_WINDOW}, Orden={SAVGOL_POLY})...")
    if len(gaze_raw_x) < SAVGOL_WINDOW:
        print(f"Error: No hay suficientes datos ({len(gaze_raw_x)}) para el filtro (ventana={SAVGOL_WINDOW}).")
        sys.exit()
    gaze_filtered_x = savgol_filter(gaze_raw_x, SAVGOL_WINDOW, SAVGOL_POLY)
    gaze_filtered_y = savgol_filter(gaze_raw_y, SAVGOL_WINDOW, SAVGOL_POLY)
    print("Filtro aplicado exitosamente.")

    # --- PASO 5: APLICAR CALIBRACIÓN DE HOMOGRAFÍA (¡EL ARREGLO!) ---
    print("\n--- PASO 4: APLICANDO CALIBRACIÓN DE HOMOGRAFÍA (CON MATRIZ 'M') ---")

    # Preparar los datos de la mirada (FILTRADOS) para la transformación
    # Formato: (N, 1, 2) como pide cv2.perspectiveTransform
    gaze_filtered_xy = np.stack((gaze_filtered_x, gaze_filtered_y), axis=1).reshape(-1, 1, 2).astype(np.float32)
    
    # Preparar los datos de la mirada (CRUDOS) para el gráfico 1
    gaze_raw_xy = np.stack((gaze_raw_x, gaze_raw_y), axis=1).reshape(-1, 1, 2).astype(np.float32)

    # Aplicar la transformación de perspectiva
    gaze_calibrated_filtered = cv2.perspectiveTransform(gaze_filtered_xy, M)
    gaze_calibrated_raw = cv2.perspectiveTransform(gaze_raw_xy, M)

    # Asignar los nuevos valores calibrados al DataFrame
    df_valid['gaze_pixel_x'] = gaze_calibrated_filtered[:, 0, 0]
    df_valid['gaze_pixel_y'] = gaze_calibrated_filtered[:, 0, 1]

    # (Para el Gráfico 1)
    gaze_calibrated_raw_x = gaze_calibrated_raw[:, 0, 0]
    gaze_calibrated_raw_y = gaze_calibrated_raw[:, 0, 1]
    print("Mapeo de homografía aplicado a los datos de la espiral.")

    # --- PASO 6: CALCULAR VELOCIDAD DE LA MIRADA (CALIBRADA) ---
    print("\n--- PASO 5: CALCULANDO VELOCIDAD DE MIRADA (CALIBRADA) ---")
    df_valid['dt_ms'] = df_valid['timestamp_ms'].diff()
    df_valid['dt_s'] = df_valid['dt_ms'] / 1000.0
    df_valid['dx_gaze'] = df_valid['gaze_pixel_x'].diff()
    df_valid['dy_gaze'] = df_valid['gaze_pixel_y'].diff()
    df_valid['distance_gaze_px'] = np.sqrt(df_valid['dx_gaze']**2 + df_valid['dy_gaze']**2)
    df_valid['velocity_gaze_px_s'] = df_valid['distance_gaze_px'] / df_valid['dt_s']
    df_valid.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned_vel = df_valid.dropna(subset=['velocity_gaze_px_s'])

    if df_cleaned_vel.empty:
        print("ADVERTENCIA: No se pudo calcular la velocidad de la mirada")
        mean_velocity_gaze = 0.0
    else:
        mean_velocity_gaze = df_cleaned_vel['velocity_gaze_px_s'].mean()
        
    print(f"Velocidad Promedio de la Mirada (CALIBRADA): {mean_velocity_gaze:.2f} px/s")

    # --- PASO 7: MOSTRAR GRÁFICO 1 (Crudo vs Filtrado) ---
    print(f"\nGenerando Gráfico 1 (Crudo vs Filtrado) en {WIDTH}x{HEIGHT}...")
    fig_calib, ax_calib = plt.subplots(figsize=(15, 9))
    ax_calib.set_facecolor('white')
    # Dibujar trayectoria CRUDA (gris, debajo)
    ax_calib.plot(gaze_calibrated_raw_x, gaze_calibrated_raw_y, color='gray', 
            lw=1, alpha=0.6, label=f'Trayectoria Cruda (Calibrada)')
    # Dibujar trayectoria FILTRADA (verde, encima)
    ax_calib.plot(df_valid['gaze_pixel_x'], df_valid['gaze_pixel_y'], color='green', 
            lw=1.5, alpha=0.8, label=f'Trayectoria Filtrada (Calibrada) (N={len(df_valid)})')
    
    start_gaze_x = df_valid['gaze_pixel_x'].iloc[0]
    start_gaze_y = df_valid['gaze_pixel_y'].iloc[0]
    end_gaze_x = df_valid['gaze_pixel_x'].iloc[-1]
    end_gaze_y = df_valid['gaze_pixel_y'].iloc[-1]
    ax_calib.scatter([start_gaze_x], [start_gaze_y], 
               s=200, facecolors='green', edgecolors='k', zorder=10, label='Inicio Mirada')
    ax_calib.text(start_gaze_x, start_gaze_y - 15, 'INICIO', color='green', ha='center', weight='bold')
    ax_calib.scatter([end_gaze_x], [end_gaze_y], 
               s=200, facecolors='red', edgecolors='k', zorder=10, label='Fin Mirada')
    ax_calib.text(end_gaze_x, end_gaze_y + 25, 'FIN', color='red', ha='center', weight='bold')

    ax_calib.set_xlim(0, WIDTH)
    ax_calib.set_ylim(0, HEIGHT)
    ax_calib.invert_yaxis()
    ax_calib.set_title("Gráfico 1: Crudo vs. Filtrado (Calibración K-Means + Homografía)")
    ax_calib.set_xlabel("Píxeles X")
    ax_calib.set_ylabel("Píxeles Y")
    ax_calib.legend()
    ax_calib.grid(True, linestyle='--', alpha=0.5)
    ax_calib.set_aspect('equal')
    print("Mostrando Gráfico 1. Cierra esta ventana para continuar...")
    plt.show()

    # --- PASO 8: SINCRONIZAR Y CALCULAR ERROR ---
    print("\n--- PASO 6: CALCULANDO ERROR DE SEGUIMIENTO DINÁMICO ---")
    errores_en_pixeles = []
    for _, row in df_valid.iterrows():
        tiempo_seg = row['timestamp_ms'] / 1000.0
        frame_index = int(tiempo_seg * FPS)
        
        if frame_index >= len(ground_truth_path):
            break
            
        truth_x, truth_y = ground_truth_path[frame_index]
        gaze_x = row['gaze_pixel_x']
        gaze_y = row['gaze_pixel_y']
        
        error_px = math.hypot(truth_x - gaze_x, truth_y - gaze_y)
        
        # Filtro de error
        if error_px < (WIDTH / 3): # Un filtro razonable
            errores_en_pixeles.append(error_px)

    # --- PASO 9: REPORTAR RESULTADOS ---
    if not errores_en_pixeles:
        print("No se pudieron calcular errores. ¿Datos vacíos o mala sincronización?")
        sys.exit()

    mean_error = np.mean(errores_en_pixeles)
    std_dev_error = np.std(errores_en_pixeles)
    print("\n--- REPORTE DE PRECISIÓN (K-Means + Homografía + Sav-Gol) ---")
    print(f"Error Promedio de Seguimiento: {mean_error:.2f} píxeles")
    print(f"Desviación (Jitter) Promedio: {std_dev_error:.2f} píxeles")
    print("--------------------------------------------------")

    # REPORTE DE VELOCIDAD
    print("\n--- REPORTE DE VELOCIDAD (K-Means + Homografía + Sav-Gol) ---")
    print(f"Velocidad Promedio del Estímulo: {mean_velocity_truth:.2f} píxeles/s")
    print(f"Velocidad Promedio de la Mirada: {mean_velocity_gaze:.2f} píxeles/s")

    difference_v = mean_velocity_gaze - mean_velocity_truth
    percentage_diff_v = (difference_v / mean_velocity_truth) * 100

    print(f"Diferencia (Gaze - Estímulo): {difference_v:.2f} px/s")
    if difference_v > 0:
        print(f"La mirada fue un {percentage_diff_v:.1f}% más RÁPIDA que el estímulo.")
    else:
        print(f"La mirada fue un {abs(percentage_diff_v):.1f}% más LENTA que el estímulo.")
    print("---------------------------------")

    # --- PASO 10: GRÁFICO 2 (Final) ---
    print(f"\nGenerando Gráfico 2 (Mirada CALIBRADA vs Verdad Absoluta) en {WIDTH}x{HEIGHT}...")
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_facecolor('white')
    # Dibujar la trayectoria "Verdadera" (rojo)
    ax.plot(interp_x, interp_y, color='red', lw=4, 
            label=f'Trayectoria Real (Target) (N={len(ground_truth_path)})')
    # Dibujar la trayectoria de la "Mirada" (verde, CALIBRADA)
    ax.plot(df_valid['gaze_pixel_x'], df_valid['gaze_pixel_y'], color='green', 
            lw=1.5, alpha=0.7, label=f'Trayectoria Calibrada (K-Means Homografía) (N={len(df_valid)})')
    
    ax.scatter([start_gaze_x], [start_gaze_y], 
               s=200, facecolors='green', edgecolors='k', zorder=10, label='Inicio Mirada')
    ax.text(start_gaze_x, start_gaze_y - 15, 'INICIO', color='green', ha='center', weight='bold')
    ax.scatter([end_gaze_x], [end_gaze_y], 
               s=200, facecolors='red', edgecolors='k', zorder=10, label='Fin Mirada')
    ax.text(end_gaze_x, end_gaze_y + 25, 'FIN', color='red', ha='center', weight='bold')

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.invert_yaxis()
    ax.set_title(f"Gráfico 2: Validación (Error Promedio CALIBRADO: {mean_error:.2f} px)")
    ax.set_xlabel("Píxeles X")
    ax.set_ylabel("Píxeles Y")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    print("Mostrando Gráfico 2...")
    plt.show()

    print("\nAnálisis completado.")


# --- Ejecutar el script principal ---
if __name__ == "__main__":
    # Asegurarse de que las librerías necesarias estén instaladas
    try:
        import pandas
        import numpy
        import pygame
        import matplotlib
        import scipy
        import cv2
        import sklearn
    except ImportError as e:
        print(f"Error: Falta una librería necesaria: {e.name}")
        print(f"Por favor, instala la librería con: pip install {e.name}")
        if e.name == 'cv2':
            print("Para OpenCV, usa: pip install opencv-python")
        if e.name == 'sklearn':
            print("Para K-Means, usa: pip install scikit-learn")
        sys.exit(1)
        
    main_analysis()