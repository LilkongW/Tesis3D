import pandas as pd
import numpy as np
import cv2
import os
import sys
from sklearn.cluster import KMeans
import pygame # Para obtener la resolución de pantalla
import math
import matplotlib.pyplot as plt

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- 1. CONFIGURACIÓN ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# ¡SOLO EDITA ESTAS DOS LÍNEAS!
# (1 = Experimento 9-puntos, 2 = Experimento 5-puntos)
TIPO_EXPERIMENTO = 2

# Ruta al archivo CSV de entrada
INPUT_CSV_PATH = r"/home/vit/Documentos/Tesis3D/Data/Experimento_2/Victor_data/Victor_intento_1_data.csv"

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- FIN DE LA CONFIGURACIÓN ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def calcular_puntos_reales_pantalla(tipo_experimento):
    """
    Usa Pygame para obtener la resolución de pantalla y calcular
    las coordenadas de los centros de la cuadrícula según el experimento.
    """
    print("Obteniendo resolución de pantalla con Pygame...")
    pygame.init()
    display_info = pygame.display.Info()
    WIDTH, HEIGHT = display_info.current_w, display_info.current_h
    pygame.quit() # Salir de pygame, solo necesitábamos la info
    
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
                print(f"  Punto ({r},{c}): Píxel ({x}, {y})")
                
    elif tipo_experimento == 2:
        print("Calculando 5 puntos de calibración reales (Exp 2)...")
        # (Lógica basada en tu script Crear_animaciones.py)
        
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
        for i, p in enumerate(puntos_reales):
            print(f"  Punto {i+1}: Píxel ({p[0]}, {p[1]})")

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
    
    print(f"Buscando los {n_clusters} centros de fijación 'torcidos' (K-Means)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(gaze_data)
    
    centros_gaze = kmeans.cluster_centers_
    puntos_gaze_ordenados = []
    
    # --- LÓGICA DE ORDENAMIENTO DINÁMICA ---
    # Asumimos Eje Gaze_Y: "Arriba" (Positivo) -> "Abajo" (Negativo)
    # Asumimos Eje Gaze_X: "Izquierda" (Negativo) -> "Derecha" (Positivo)
    
    if tipo_experimento == 1:
        print("Ordenando centros para 3x3 (Exp 1)...")
        # (Tu lógica original: Y desciende, X desciende)
        # NOTA: Tu script original ordenaba Y descendente (arriba) y X descendente (izquierda).
        # Lo cambio a X ascendente (izquierda a derecha) para que coincida con el orden de puntos (1,2,3... 7,8,9)
        
        indices_y_ordenados = np.argsort(-centros_gaze[:, 1]) # Y descendente (filas 0, 1, 2)
        
        filas = [
            centros_gaze[indices_y_ordenados[0:3]], # Fila superior
            centros_gaze[indices_y_ordenados[3:6]], # Fila media
            centros_gaze[indices_y_ordenados[6:9]]  # Fila inferior
        ]
        
        for fila in filas:
            indices_x_ordenados_fila = np.argsort(fila[:, 0]) # X ascendente (col 0, 1, 2)
            puntos_gaze_ordenados.extend(fila[indices_x_ordenados_fila])
            
    elif tipo_experimento == 2:
        print("Ordenando centros para 5-puntos (Exp 2)...")
        # Esta lógica debe replicar el orden de la animación:
        # TL, TR, BL, BR, C
        
        # Ordenar todos por Y (arriba-abajo)
        centros_ordenados_y = centros_gaze[np.argsort(-centros_gaze[:, 1])] # Y descendente (arriba primero)
        
        # Separar los 2 de arriba, 2 de abajo, 1 de centro
        # (Esto asume que el centro está bien separado de las esquinas)
        if len(centros_ordenados_y) != 5:
             print("Error: K-Means no devolvió 5 centros como se esperaba.")
             return None, None
             
        top_2 = centros_ordenados_y[0:2]
        center_1 = centros_ordenados_y[2:3] # Temporalmente
        bottom_2 = centros_ordenados_y[3:5]
        
        # Re-evaluar: Es más robusto buscar el centro por X
        # Ordenar por Y, tomar los 2 de arriba y 2 de abajo. El que sobra es el centro.
        
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
        
    print(f"Centros de Gaze 'torcidos' ({n_clusters} encontrados y ordenados):")
    for i, p in enumerate(puntos_gaze_ordenados):
        print(f"  Punto {i+1}: ({p[0]:.3f}, {p[1]:.3f})")

    return np.array(puntos_gaze_ordenados, dtype=np.float32), df_valid

def mapear_trayectoria(csv_path, tipo_experimento):
    
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
    if puntos_gaze_ordenados is None:
        return

    # Verificar que las longitudes coincidan
    if len(puntos_pantalla_np) != len(puntos_gaze_ordenados):
        print("¡Error crítico! El número de puntos reales no coincide con los clusters de gaze encontrados.")
        print(f"Puntos reales: {len(puntos_pantalla_np)}, Clusters gaze: {len(puntos_gaze_ordenados)}")
        return

    # --- 3. Calcular la Matriz de Homografía (¡CORREGIDO!) ---
    print("Calculando matriz de calibración por Homografía (method=0)...")
    # H es la matriz que transforma puntos_gaze -> puntos_pantalla
    H, _ = cv2.findHomography(puntos_gaze_ordenados, puntos_pantalla_np, method=0)
    
    if H is None:
        print("Error: No se pudo calcular la matriz de homografía.")
        return

    # --- 4. Transformar los N Centros Gaze para medir el error (¡CORREGIDO!) ---
    print("Transformando centros de gaze para calcular el error...")
    
    # Reformatear para cv2.perspectiveTransform: (1, N, 2)
    puntos_gaze_cv2 = np.expand_dims(puntos_gaze_ordenados, axis=0)
    
    # Aplicar la transformación de perspectiva
    puntos_gaze_mapeados_cv2 = cv2.perspectiveTransform(puntos_gaze_cv2, H)
    
    # Volver a formatear a (N, 2)
    puntos_gaze_mapeados = puntos_gaze_mapeados_cv2[0].astype(np.float32)

    # --- 5. Calcular Error (Precisión) ---
    print(f"\n--- REPORTE DE PRECISIÓN DE CALIBRACIÓN ({n_clusters} Puntos) ---")
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
        ax.text(puntos_pantalla_np[i, 0], puntos_pantalla_np[i, 1] + 30, str(i+1), 
                color='black', ha='center', va='center', fontsize=10, weight='bold')

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
        # Llamar a la función principal con la configuración
        mapear_trayectoria(INPUT_CSV_PATH, TIPO_EXPERIMENTO)