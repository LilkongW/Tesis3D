import pandas as pd
import numpy as np
import cv2
import os
import sys
from sklearn.cluster import KMeans
import pygame # Para obtener la resolución de pantalla
import math
import matplotlib.pyplot as plt
import seaborn as sns

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- 1. CONFIGURACIÓN (¡SOLO EDITA LA RUTA DEL CSV!) ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Ruta al archivo CSV de entrada (el de 9 puntos)
INPUT_CSV_PATH = "/home/vit/Documentos/Tesis3D/Data/Experimento_1/Victor_data/Victor3_intento_1_data.csv"

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- FIN DE LA CONFIGURACIÓN ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def calcular_puntos_reales_pantalla():
    """
    Usa Pygame para obtener la resolución de pantalla y calcular
    las 9 coordenadas de los centros de la cuadrícula.
    """
    print("Obteniendo resolución de pantalla con Pygame...")
    pygame.init()
    display_info = pygame.display.Info()
    WIDTH, HEIGHT = display_info.current_w, display_info.current_h
    pygame.quit()
    print(f"Resolución detectada: {WIDTH}x{HEIGHT}")

    ROWS, COLS = 3, 3
    cell_width = WIDTH // COLS
    cell_height = HEIGHT // ROWS
    
    puntos_reales = []
    print("Calculando 9 puntos de calibración reales:")
    for r in range(ROWS):
        for c in range(COLS):
            x = (c * cell_width) + (cell_width // 2)
            y = (r * cell_height) + (cell_height // 2)
            puntos_reales.append((x, y))
            
    return np.array(puntos_reales, dtype=np.float32), (WIDTH, HEIGHT)

def encontrar_centros_de_gaze(csv_path):
    """
    Carga los datos crudos (gaze_x, gaze_y) y usa K-Means
    para encontrar los 9 centros de fijación.
    """
    print(f"Cargando datos de mirada desde: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None, None
    
    df_valid = df[df['valid_deteccion'] == True].copy()
    if df_valid.empty:
        return None, None

    # Usar los datos CRUDOS (torcidos)
    gaze_data = df_valid[['gaze_x', 'gaze_y']].values
    
    print("Buscando los 9 centros de fijación 'torcidos' (K-Means)...")
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    kmeans.fit(gaze_data)
    centros_gaze = kmeans.cluster_centers_
    
    # Ordenar los centros (Y descendente, X descendente)
    indices_y_ordenados = np.argsort(-centros_gaze[:, 1])
    filas = [
        centros_gaze[indices_y_ordenados[0:3]],
        centros_gaze[indices_y_ordenados[3:6]],
        centros_gaze[indices_y_ordenados[6:9]]
    ]
    puntos_gaze_ordenados = []
    for fila in filas:
        indices_x_ordenados_fila = np.argsort(-fila[:, 0])
        puntos_gaze_ordenados.extend(fila[indices_x_ordenados_fila])
        
    return np.array(puntos_gaze_ordenados, dtype=np.float32), df_valid

def calibrar_y_plotear(csv_path):
    
    # --- 1. Obtener Puntos de Destino (Pantalla) ---
    puntos_pantalla_np, screen_size = calcular_puntos_reales_pantalla()
    SCREEN_WIDTH, SCREEN_HEIGHT = screen_size

    # --- 2. Encontrar Puntos de Origen (Gaze) ---
    puntos_gaze_ordenados, df_valid = encontrar_centros_de_gaze(csv_path)
    if puntos_gaze_ordenados is None:
        print("No se pudieron encontrar los centros de la mirada.")
        return

    # --- 3. Calcular la Matriz de Homografía (La "Doble Corrección") ---
    print("Calculando matriz de calibración por Homografía (method=0)...")
    # H es la matriz que "destuerce" los datos crudos
    H, _ = cv2.findHomography(puntos_gaze_ordenados, puntos_pantalla_np, method=0)
    
    if H is None:
        print("Error: No se pudo calcular la matriz de homografía.")
        return

    # --- 4. Aplicar la Transformación a *TODA* la trayectoria ---
    print("Aplicando calibración a toda la trayectoria...")
    trayectoria_gaze_cruda = df_valid[['gaze_x', 'gaze_y']].values.astype(np.float32)
    trayectoria_gaze_cv2 = np.expand_dims(trayectoria_gaze_cruda, axis=0)
    
    # Aplicar la "magia"
    trayectoria_mapeada_cv2 = cv2.perspectiveTransform(trayectoria_gaze_cv2, H)
    
    # Guardar los nuevos puntos calibrados en el DataFrame
    trayectoria_mapeada_limpia = trayectoria_mapeada_cv2[0]
    df_valid['calibrated_x'] = trayectoria_mapeada_limpia[:, 0]
    df_valid['calibrated_y'] = trayectoria_mapeada_limpia[:, 1]
    
    # --- 5. Dibujar y Mostrar el Resultado (con Matplotlib) ---
    
    print("\nGenerando gráfico de heatmap y trayectoria CALIBRADOS...")
    
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_facecolor('white')
    
    # --- PASO A: Dibujar el Heatmap (Fondo) ---
    # ¡Usar los datos ya CALIBRADOS!
    sns.kdeplot(
        ax=ax,
        x=df_valid['calibrated_x'], # <-- CALIBRADO
        y=df_valid['calibrated_y'], # <-- CALIBRADO
        fill=True,
        cmap="rocket_r",
        thresh=0.05,
        bw_adjust=0.5
    )
    
    # --- PASO B: Dibujar la Trayectoria (Encima) ---
    ax.plot(
        df_valid['calibrated_x'], # <-- CALIBRADO
        df_valid['calibrated_y'], # <-- CALIBRADO
        color='lime', 
        lw=1.5, 
        alpha=0.6, 
        label='Trayectoria Calibrada'
    )
                  
    # --- PASO C: Dibujar los PUNTOS REALES (Objetivos) ---
    ax.scatter(puntos_pantalla_np[:, 0], puntos_pantalla_np[:, 1], 
               s=600, facecolors='none', edgecolors='blue', lw=2, 
               marker='o', label='Objetivo Real (Pantalla)')
    
    # Poner los números de los puntos
    for i, (px, py) in enumerate(puntos_pantalla_np.astype(int)):
        ax.text(px, py, str(i+1), 
                color='black', ha='center', va='center', fontsize=10, weight='bold')

    # --- Configurar los ejes para que coincidan con los píxeles ---
    ax.set_xlim(0, SCREEN_WIDTH)
    ax.set_ylim(0, SCREEN_HEIGHT)
    ax.invert_yaxis() # Y=0 arriba
    
    ax.set_title(f"Heatmap y Trayectoria Calibrados a Pantalla {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    ax.set_xlabel("Píxeles X")
    ax.set_ylabel("Píxeles Y")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    try:
        plt.show() # Mostrar la ventana de Matplotlib
        print("Ventana de gráfico cerrada.")
    except Exception as e:
        print(f"Error al mostrar el gráfico: {e}")
    plt.close(fig)


# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    
    # Comprobar si el archivo existe
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: No se encontró el archivo CSV en la ruta especificada:")
        print(f"{INPUT_CSV_PATH}")
    else:
        # Llamar a la función principal
        calibrar_y_plotear(INPUT_CSV_PATH)