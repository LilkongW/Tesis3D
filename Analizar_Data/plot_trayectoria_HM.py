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
# --- 1. CONFIGURACIÓN ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# ¡SOLO EDITA ESTAS DOS LÍNEAS!
# (1 = Experimento 9-puntos, 2 = Experimento 5-puntos)
TIPO_EXPERIMENTO = 1 

# Ruta al archivo CSV de entrada
INPUT_CSV_PATH = f"C:\\Users\\Victor\\Documents\\Tesis3D\\Data\\Experimento_{TIPO_EXPERIMENTO}\\Victor_data\\Victor_intento_5_data.csv"

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- FIN DE LA CONFIGURACIÓN ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def calcular_puntos_reales_pantalla(tipo_experimento):
    """
    Usa Pygame para obtener la resolución de pantalla y calcular
    las coordenadas de los centros según el experimento.
    """
    print("Obteniendo resolución de pantalla con Pygame...")
    pygame.init()
    display_info = pygame.display.Info()
    WIDTH, HEIGHT = display_info.current_w, display_info.current_h
    pygame.quit()
    print(f"Resolución detectada: {WIDTH}x{HEIGHT}")

    puntos_reales = []
    
    # Rejilla base 3x3 para los cálculos
    ROWS, COLS = 3, 3
    cell_width = WIDTH // COLS
    cell_height = HEIGHT // ROWS
    
    if tipo_experimento == 1:
        print("Calculando 9 puntos de calibración reales (Exp 1)...")
        for r in range(ROWS):
            for c in range(COLS):
                # Calcular el centro de la celda (col, row)
                x = (c * cell_width) + (cell_width // 2)
                y = (r * cell_height) + (cell_height // 2)
                puntos_reales.append((x, y))
                
    elif tipo_experimento == 2:
        print("Calculando 5 puntos de calibración reales (Exp 2)...")
        # (Lógica basada en Crear_animaciones.py)
        
        # 1. Esquina superior izquierda (fila 0, col 0)
        pos_top_left = (cell_width // 2, cell_height // 2)
        
        # 2. Esquina superior derecha (fila 0, col 2)
        pos_top_right = (2 * cell_width + cell_width // 2, cell_height // 2)
        
        # 3. Esquina inferior izquierda (fila 2, col 0)
        pos_bottom_left = (cell_width // 2, 2 * cell_height + cell_height // 2)
        
        # 4. Esquina inferior derecha (fila 2, col 2)
        pos_bottom_right = (2 * cell_width + cell_width // 2, 2 * cell_height + cell_height // 2)
        
        # 5. Centro (fila 1, col 1)
        pos_center = (cell_width + cell_width // 2, cell_height + cell_height // 2)
        
        # El orden debe coincidir con la animación:
        puntos_reales = [
            pos_top_left,
            pos_top_right,
            pos_bottom_left,
            pos_bottom_right,
            pos_center
        ]

    else:
        print(f"Error: TIPO_EXPERIMENTO {tipo_experimento} no reconocido.")
        return None, None
            
    return np.array(puntos_reales, dtype=np.float32), (WIDTH, HEIGHT)

def encontrar_centros_de_gaze(csv_path, n_clusters, tipo_experimento):
    """
    Carga los datos crudos (gaze_x, gaze_y) y usa K-Means
    para encontrar los N centros de fijación.
    Luego, los ordena según la lógica del experimento.
    """
    print(f"Cargando datos de mirada desde: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {csv_path}")
        return None, None
    
    df_valid = df[df['valid_deteccion'] == True].copy()
    if df_valid.empty:
        print("No se encontraron detecciones válidas.")
        return None, None

    # Usar los datos CRUDOS (torcidos)
    gaze_data = df_valid[['gaze_x', 'gaze_y']].values
    
    print(f"Buscando los {n_clusters} centros de fijación 'torcidos' (K-Means)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(gaze_data)
    centros_gaze = kmeans.cluster_centers_
    
    puntos_gaze_ordenados = []
    
    # --- LÓGICA DE ORDENAMIENTO DINÁMICA ---
    # Asumimos Eje Gaze_Y: "Arriba" (Positivo) -> "Abajo" (Negativo)
    
    if tipo_experimento == 1:
        print("Ordenando centros para 3x3 (Exp 1)...")
        # Ordenar por Y descendente (filas 0, 1, 2)
        indices_y_ordenados = np.argsort(-centros_gaze[:, 1]) 
        
        filas = [
            centros_gaze[indices_y_ordenados[0:3]], # Fila superior
            centros_gaze[indices_y_ordenados[3:6]], # Fila media
            centros_gaze[indices_y_ordenados[6:9]]  # Fila inferior
        ]
        
        for fila in filas:
            # Ordenar por X ascendente (col 0, 1, 2)
            indices_x_ordenados_fila = np.argsort(fila[:, 0]) 
            puntos_gaze_ordenados.extend(fila[indices_x_ordenados_fila])
            
    elif tipo_experimento == 2:
        print("Ordenando centros para 5-puntos (Exp 2)...")
        # Lógica para replicar el orden: TL, TR, BL, BR, C
        
        # 1. Ordenar por Y descendente (arriba primero)
        centros_ordenados_y = centros_gaze[np.argsort(-centros_gaze[:, 1])] 
        top_2_candidatos = centros_ordenados_y[:2]
        bottom_2_candidatos = centros_ordenados_y[-2:]
        
        # 2. Encontrar el centro (el que no está en top ni bottom)
        candidatos_esquinas = np.concatenate((top_2_candidatos, bottom_2_candidatos))
        centro_gaze = None
        for centro in centros_gaze:
            if not np.any(np.all(centro == candidatos_esquinas, axis=1)):
                centro_gaze = centro
                break
        
        if centro_gaze is None:
             print("Advertencia: No se pudo aislar el punto central. Usando el punto medio de Y.")
             centro_gaze = centros_ordenados_y[2] # Fallback
        
        # 3. Ordenar los 2 de arriba por X (izquierda-derecha)
        top_2_ordenados = top_2_candidatos[np.argsort(top_2_candidatos[:, 0])] # X ascendente
        
        # 4. Ordenar los 2 de abajo por X
        bottom_2_ordenados = bottom_2_candidatos[np.argsort(bottom_2_candidatos[:, 0])] # X ascendente
        
        # 5. Reconstruir en el orden de la animación
        puntos_gaze_ordenados = [
            top_2_ordenados[0],   # Top-Left
            top_2_ordenados[1],   # Top-Right
            bottom_2_ordenados[0], # Bottom-Left
            bottom_2_ordenados[1], # Bottom-Right
            centro_gaze           # Center
        ]

    else:
        print(f"Error: Lógica de ordenamiento no definida para tipo {tipo_experimento}")
        return None, None
        
    return np.array(puntos_gaze_ordenados, dtype=np.float32), df_valid

def calibrar_y_plotear(csv_path, tipo_experimento):
    
    # --- 0. Determinar N de Clusters ---
    if tipo_experimento == 1:
        n_clusters = 9
    elif tipo_experimento == 2:
        n_clusters = 5
    else:
        print(f"Error: TIPO_EXPERIMENTO {tipo_experimento} no es válido.")
        return

    # --- 1. Obtener Puntos de Destino (Pantalla) ---
    puntos_pantalla_np, screen_size = calcular_puntos_reales_pantalla(tipo_experimento)
    if puntos_pantalla_np is None:
        return
    SCREEN_WIDTH, SCREEN_HEIGHT = screen_size

    # --- 2. Encontrar Puntos de Origen (Gaze) ---
    puntos_gaze_ordenados, df_valid = encontrar_centros_de_gaze(csv_path, n_clusters, tipo_experimento)
    if puntos_gaze_ordenados is None or df_valid is None:
        print("No se pudieron encontrar los centros de la mirada o no hay datos válidos.")
        return
        
    if len(puntos_pantalla_np) != len(puntos_gaze_ordenados):
        print("¡Error crítico! El número de puntos reales no coincide con los clusters de gaze encontrados.")
        return

    # --- 3. Calcular y Guardar la Matriz de Homografía ---
    print("Calculando matriz de calibración por Homografía (method=0)...")
    H, _ = cv2.findHomography(puntos_gaze_ordenados, puntos_pantalla_np, method=0)
    
    if H is None:
        print("Error: No se pudo calcular la matriz de homografía.")
        return

    # --- ¡NUEVO: Guardar la matriz H en un archivo .npy! ---
    output_matrix_path = 'matriz_calibracion.npy'
    try:
        np.save(output_matrix_path, H)
        print(f"\n✅ Matriz de calibración guardada en: {output_matrix_path}")
    except Exception as e:
        print(f"\n🚨 Error: No se pudo guardar la matriz de calibración. {e}")
    # --- Fin de la sección de guardado ---

    # --- 4. Calcular Error de Precisión ---
    print("\nCalculando precisión de la calibración...")
    
    # Transformar los centros de gaze "torcidos" a píxeles de pantalla
    puntos_gaze_cv2 = np.expand_dims(puntos_gaze_ordenados, axis=0)
    puntos_gaze_mapeados_cv2 = cv2.perspectiveTransform(puntos_gaze_cv2, H)
    puntos_gaze_mapeados = puntos_gaze_mapeados_cv2[0]
    
    distancias_error = []
    for i in range(n_clusters):
        punto_real = puntos_pantalla_np[i]
        punto_medido = puntos_gaze_mapeados[i]
        
        # Calcular la distancia Euclidiana (error en píxeles)
        distancia = math.hypot(punto_real[0] - punto_medido[0], punto_real[1] - punto_medido[1])
        distancias_error.append(distancia)
        
    mae = np.mean(distancias_error)
    std_dev = np.std(distancias_error)
    
    print("-------------------------------------------------")
    print("--- REPORTE DE PRECISIÓN DE CALIBRACIÓN ---")
    print(f"  Error Absoluto Medio (MAE):  {mae:.2f} píxeles")
    print(f"  Desviación Estándar (Error): {std_dev:.2f} píxeles")
    print("-------------------------------------------------")

    # --- 5. Aplicar la Transformación a *TODA* la trayectoria ---
    print("\nAplicando calibración a toda la trayectoria...")
    trayectoria_gaze_cruda = df_valid[['gaze_x', 'gaze_y']].values.astype(np.float32)
    trayectoria_gaze_cv2 = np.expand_dims(trayectoria_gaze_cruda, axis=0)
    
    trayectoria_mapeada_cv2 = cv2.perspectiveTransform(trayectoria_gaze_cv2, H)
    
    trayectoria_mapeada_limpia = trayectoria_mapeada_cv2[0]
    df_valid['calibrated_x'] = trayectoria_mapeada_limpia[:, 0]
    df_valid['calibrated_y'] = trayectoria_mapeada_limpia[:, 1]
    
    # --- 6. Dibujar y Mostrar el Resultado (con Matplotlib) ---
    
    print("\nGenerando gráfico de heatmap y trayectoria CALIBRADOS...")
    
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_facecolor('white')
    
    # --- PASO A: Dibujar el Heatmap (Fondo) ---
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
    
    # --- PASO D: (Opcional) Dibujar los centros de gaze medidos ---
    ax.scatter(puntos_gaze_mapeados[:, 0], puntos_gaze_mapeados[:, 1], 
               s=600, marker='x', color='cyan', lw=2, label='Centro Gaze Calibrado (X)')
    
    # Poner los números de los puntos
    for i, (px, py) in enumerate(puntos_pantalla_np.astype(int)):
        ax.text(px, py, str(i+1), 
                color='black', ha='center', va='center', fontsize=10, weight='bold')

    # --- Configurar los ejes para que coincidan con los píxeles ---
    ax.set_xlim(0, SCREEN_WIDTH)
    ax.set_ylim(0, SCREEN_HEIGHT)
    ax.invert_yaxis() # Y=0 arriba
    
    # Título ahora incluye el error
    ax.set_title(f"Heatmap Calibrado (Error Promedio: {mae:.2f} px) - Pantalla {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
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
        calibrar_y_plotear(INPUT_CSV_PATH, TIPO_EXPERIMENTO)