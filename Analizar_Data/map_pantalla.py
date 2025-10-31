import pandas as pd
import numpy as np
import cv2
import os
import sys
from sklearn.cluster import KMeans
import pygame # Para obtener la resolución de pantalla
import math
import matplotlib.pyplot as plt # ¡NUEVA IMPORTACIÓN!

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- 1. CONFIGURACIÓN (¡SOLO EDITA LA RUTA DEL CSV!) ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Ruta al archivo CSV de entrada (el de 9 puntos, como Victor3_intento_1_data.csv)
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
    pygame.quit() # Salir de pygame, solo necesitábamos la info
    
    print(f"Resolución detectada: {WIDTH}x{HEIGHT}")

    ROWS, COLS = 3, 3
    cell_width = WIDTH // COLS
    cell_height = HEIGHT // ROWS
    
    puntos_reales = []
    print("Calculando 9 puntos de calibración reales:")
    for r in range(ROWS):
        for c in range(COLS):
            # Calcular el centro de la celda (col, row)
            x = (c * cell_width) + (cell_width // 2)
            y = (r * cell_height) + (cell_height // 2)
            puntos_reales.append((x, y))
            print(f"  Punto ({r},{c}): Píxel ({x}, {y})")
            
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
        print(f"Error: No se encontró el archivo en {csv_path}")
        return None
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return None

    # Filtrar solo por detecciones válidas
    df_valid = df[df['valid_deteccion'] == True].copy()
    if df_valid.empty:
        print("No se encontraron detecciones válidas.")
        return None

    # --- ¡IMPORTANTE! Usar los datos CRUDOS ---
    gaze_data = df_valid[['gaze_x', 'gaze_y']].values
    
    print("Buscando los 9 centros de fijación 'torcidos' (K-Means)...")
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    kmeans.fit(gaze_data)
    
    centros_gaze = kmeans.cluster_centers_
    
    # --- Ordenar los centros ---
    # (Como en tu terminal: Y desciende, X desciende)
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
        
    print("Centros de Gaze 'torcidos' (encontrados y ordenados):")
    for i, p in enumerate(puntos_gaze_ordenados):
        print(f"  Punto {i+1}: ({p[0]:.3f}, {p[1]:.3f})")

    return np.array(puntos_gaze_ordenados, dtype=np.float32), df_valid

def mapear_trayectoria(csv_path):
    
    # --- 1. Obtener Puntos de Destino (Pantalla) ---
    puntos_pantalla_np, screen_size = calcular_puntos_reales_pantalla()
    SCREEN_WIDTH, SCREEN_HEIGHT = screen_size

    # --- 2. Encontrar Puntos de Origen (Gaze) ---
    puntos_gaze_ordenados, df_valid = encontrar_centros_de_gaze(csv_path)
    if puntos_gaze_ordenados is None:
        return

    # --- 3. Calcular la Matriz de Homografía (¡CORREGIDO!) ---
    print("Calculando matriz de calibración por Homografía (method=0)...")
    # H es la matriz que transforma puntos_gaze -> puntos_pantalla
    # ¡method=0 usa TODOS los puntos (Least Squares), no RANSAC!
    H, _ = cv2.findHomography(puntos_gaze_ordenados, puntos_pantalla_np, method=0)
    
    if H is None:
        print("Error: No se pudo calcular la matriz de homografía.")
        return

    # --- 4. Transformar los 9 Centros Gaze para medir el error (¡CORREGIDO!) ---
    print("Transformando centros de gaze para calcular el error...")
    
    # Reformatear para cv2.perspectiveTransform: (1, 9, 2)
    puntos_gaze_cv2 = np.expand_dims(puntos_gaze_ordenados, axis=0)
    
    # Aplicar la transformación de perspectiva
    puntos_gaze_mapeados_cv2 = cv2.perspectiveTransform(puntos_gaze_cv2, H)
    
    # Volver a formatear a (9, 2)
    puntos_gaze_mapeados = puntos_gaze_mapeados_cv2[0].astype(np.float32)

    # --- 5. Calcular Error (Precisión) ---
    print("\n--- REPORTE DE PRECISIÓN DE CALIBRACIÓN (HOMOGRAFÍA) ---")
    distancias_error = []
    for i in range(len(puntos_pantalla_np)):
        punto_real = puntos_pantalla_np[i]
        punto_medido = puntos_gaze_mapeados[i]
        
        distancia = math.hypot(punto_real[0] - punto_medido[0], punto_real[1] - punto_medido[1])
        distancias_error.append(distancia)
        
        print(f"  Punto {i+1} [{int(punto_real[0])}, {int(punto_real[1])}]:")
        print(f"    Mirada calibrada en [{int(punto_medido[0])}, {int(punto_medido[1])}]")
        print(f"    -> Error (Diferencia): {distancia:.2f} píxeles")

    mae = np.mean(distancias_error)
    print("-------------------------------------------------")
    print(f"PRECISIÓN GENERAL (Error Absoluto Medio): {mae:.2f} píxeles")
    print("-------------------------------------------------")

    # --- 6. Aplicar la Transformación a *TODA* la trayectoria (¡CORREGIDO!) ---
    print("Aplicando calibración a toda la trayectoria para visualización...")
    trayectoria_gaze = df_valid[['gaze_x', 'gaze_y']].values.astype(np.float32)
    trayectoria_gaze_cv2 = np.expand_dims(trayectoria_gaze, axis=0)
    
    # Usar perspectiveTransform
    trayectoria_mapeada_cv2 = cv2.perspectiveTransform(trayectoria_gaze_cv2, H)
    trayectoria_mapeada = trayectoria_mapeada_cv2[0].astype(np.float32)
    
    # --- 7. Dibujar y Mostrar el Resultado (con Matplotlib) ---
    
    print("\nMostrando resultado con Matplotlib...")
    print("(O) = Punto Real, (X) = Mirada Medida")
    
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_facecolor('white')
    
    # Dibujar la trayectoria mapeada (verde)
    ax.plot(trayectoria_mapeada[:, 0], trayectoria_mapeada[:, 1], 
            color='green', lw=1.5, alpha=0.6, label='Trayectoria Calibrada')
                  
    # Dibujar los PUNTOS REALES (Objetivos)
    ax.scatter(puntos_pantalla_np[:, 0], puntos_pantalla_np[:, 1], 
               s=600, facecolors='none', edgecolors='red', lw=2, label='Objetivo Real (O)')
               
    # Dibujar los CENTROS DE MIRADA (Medidos)
    ax.scatter(puntos_gaze_mapeados[:, 0], puntos_gaze_mapeados[:, 1], 
               s=600, marker='x', color='blue', lw=2, label='Mirada Medida (X)')
    
    # Poner los números de los puntos
    for i in range(len(puntos_pantalla_np)):
        ax.text(puntos_pantalla_np[i, 0], puntos_pantalla_np[i, 1] + 10, str(i+1), 
                color='black', ha='center', va='center', fontsize=10)

    # Configurar los ejes para que coincidan con los píxeles
    ax.set_xlim(0, SCREEN_WIDTH)
    ax.set_ylim(0, SCREEN_HEIGHT)
    ax.invert_yaxis() # Y=0 arriba
    
    ax.set_title(f"Reporte de Precisión (Error Promedio: {mae:.2f} px)")
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
        print("Error: No se encontró el archivo CSV en la ruta especificada:")
        print(f"{INPUT_CSV_PATH}")
    else:
        # Llamar a la función principal
        mapear_trayectoria(INPUT_CSV_PATH)