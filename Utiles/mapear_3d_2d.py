import pandas as pd
import numpy as np
import cv2
import os
import sys
from sklearn.cluster import KMeans
import pygame # Para obtener la resolución de pantalla
import math

# --- FUNCIÓN 1: CALCULAR LOS PUNTOS DE LA PANTALLA ---
def calcular_puntos_reales_pantalla():
    """
    Usa Pygame para obtener la resolución de pantalla y calcular
    las 9 coordenadas de los centros de la cuadrícula.
    
    Devuelve: (puntos_reales_np, (ANCHO, ALTO))
    """
    print("Obteniendo resolución de pantalla con Pygame...")
    try:
        pygame.init()
        display_info = pygame.display.Info()
        WIDTH, HEIGHT = display_info.current_w, display_info.current_h
        pygame.quit()
    except Exception as e:
        print(f"Error al iniciar Pygame (¿entorno sin cabeza?): {e}")
        print("Usando resolución por defecto 1920x1080.")
        WIDTH, HEIGHT = 1920, 1080
    
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

# --- FUNCIÓN 2: ENCONTRAR LOS PUNTOS DE LA MIRADA ---
def encontrar_centros_de_gaze(csv_path):
    """
    Carga los datos crudos (gaze_x, gaze_y) y usa K-Means
    para encontrar los 9 centros de fijación.
    
    Devuelve: (puntos_gaze_ordenados_np)
    """
    print(f"Cargando datos de mirada desde: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {csv_path}")
        return None
    
    df_valid = df[df['valid_deteccion'] == True].copy()
    if df_valid.empty:
        print("No se encontraron detecciones válidas.")
        return None

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
        
    return np.array(puntos_gaze_ordenados, dtype=np.float32)

# --- FUNCIÓN 3: OBTENER LA MATRIZ DE CALIBRACIÓN ---
def obtener_matriz_calibracion(csv_path_calibracion):
    """
    Función principal de calibración. Llama a esta función UNA VEZ
    al inicio de tu aplicación.
    
    Devuelve:
    H (Matriz de Homografía 3x3) o None si falla
    (ANCHO, ALTO) de la pantalla o None si falla
    """
    puntos_pantalla_np, screen_size = calcular_puntos_reales_pantalla()
    puntos_gaze_ordenados = encontrar_centros_de_gaze(csv_path_calibracion)
    
    if puntos_gaze_ordenados is None:
        print("Fallo al encontrar los centros de la mirada. Calibración fallida.")
        return None, None
        
    print("Calculando matriz de calibración por Homografía (method=0)...")
    # H es la matriz que "destuerce" los puntos crudos (gaze)
    # para que coincidan con los puntos reales (pantalla)
    H, _ = cv2.findHomography(puntos_gaze_ordenados, puntos_pantalla_np, method=0)
    
    if H is None:
        print("Error: No se pudo calcular la matriz de homografía.")
        return None, None
        
    print("¡Matriz de calibración 'H' calculada exitosamente!")
    return H, screen_size

# --- ¡ESTA ES LA FUNCIÓN QUE PEDISTE! ---
def mapear_mirada_a_pantalla(gaze_x, gaze_y, H):
    """
    Mapea un único vector de mirada (gaze_x, gaze_y) crudo a 
    coordenadas de píxeles (pix_x, pix_y) usando la matriz de calibración H.
    
    Devuelve:
    (pix_x, pix_y) o (None, None) si falla.
    """
    try:
        # Crear el punto en el formato que OpenCV espera: (1, 1, 2)
        punto_gaze_crudo = np.array([[[gaze_x, gaze_y]]], dtype=np.float32)
        
        # Aplicar la transformación de perspectiva
        punto_mapeado_cv2 = cv2.perspectiveTransform(punto_gaze_crudo, H)
        
        # Extraer el resultado y redondear a píxeles
        pix_x = int(round(punto_mapeado_cv2[0][0][0]))
        pix_y = int(round(punto_mapeado_cv2[0][0][1]))
        
        return (pix_x, pix_y)
    except Exception as e:
        print(f"Error al mapear punto: {e}")
        return (None, None)


# --- PUNTO DE ENTRADA / EJEMPLO DE CÓMO USARLO ---
if __name__ == "__main__":
    
    # Ruta al archivo CSV de calibración
    INPUT_CSV_PATH = "/home/vit/Documentos/Tesis3D/Data/Experimento_1/Victor_data/Victor3_intento_1_data.csv"
    
    # 1. CALIBRAR (Hacer esto una vez)
    # H es la "llave mágica"
    H, (SCREEN_WIDTH, SCREEN_HEIGHT) = obtener_matriz_calibracion(INPUT_CSV_PATH)
    
    if H is not None:
        print("\n--- ¡CALIBRACIÓN EXITOSA! ---")
        print("Probando la función de mapeo con los puntos de entrenamiento...")
        
        # --- 2. MAPEAR (Hacer esto en tiempo real) ---
        
        # Prueba 1: Punto 1 (arriba-izquierda)
        # (Valores crudos de tu terminal)
        g_x, g_y = (0.434, 0.527)
        p_x, p_y = mapear_mirada_a_pantalla(g_x, g_y, H)
        print(f"  Punto Crudo 1 ({g_x:.3f}, {g_y:.3f}) -> Mapeado a ({p_x}, {p_y}). (Esperado: ~320, 180)")

        # Prueba 2: Punto 5 (centro)
        g_x, g_y = (0.343, 0.443)
        p_x, p_y = mapear_mirada_a_pantalla(g_x, g_y, H)
        print(f"  Punto Crudo 5 ({g_x:.3f}, {g_y:.3f}) -> Mapeado a ({p_x}, {p_y}). (Esperado: ~960, 540)")

        # Prueba 3: Punto 9 (abajo-derecha)
        g_x, g_y = (0.181, 0.340)
        p_x, p_y = mapear_mirada_a_pantalla(g_x, g_y, H)
        print(f"  Punto Crudo 9 ({g_x:.3f}, {g_y:.3f}) -> Mapeado a ({p_x}, {p_y}). (Esperado: ~1600, 900)")
        
        # Prueba 4: Un punto de mirada aleatorio
        g_x, g_y = (0.3, 0.5)
        p_x, p_y = mapear_mirada_a_pantalla(g_x, g_y, H)
        print(f"  Punto Aleatorio ({g_x:.3f}, {g_y:.3f}) -> Mapeado a ({p_x}, {p_y})")

    else:
        print("\n--- CALIBRACIÓN FALLIDA ---")
        print("No se pudo generar la matriz de mapeo.")