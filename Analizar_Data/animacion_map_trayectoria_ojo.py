import pandas as pd
import numpy as np
import pygame 
import math
import sys 

# --- 1. CONFIGURACIÓN ---
ANIMATION_FPS = 60.0 # <-- CAMBIO 1: Bucle principal a 60 FPS
TRUTH_FPS = 30.0     # FPS del estímulo (punto rojo)
# (Offset y Timestamps del CSV se ignorarán)

# ¡Ruta al CSV de la espiral que queremos analizar!
SPIRAL_CSV_PATH = "/home/vit/Documentos/Tesis3D/Data/Experimento_3/Victor_data/Victor_intento_1_data.csv"

# --- 2. OBTENER DIMENSIONES DE PANTALLA ---
print("Obteniendo resolución de pantalla con Pygame...")
try:
    pygame.init() # ¡Inicializar pygame aquí!
    display_info = pygame.display.Info()
    WIDTH, HEIGHT = display_info.current_w, display_info.current_h
    # No salir de pygame todavía
    print(f"Resolución detectada: {WIDTH}x{HEIGHT}")
except Exception as e:
    print(f"Error al iniciar Pygame (¿entorno sin cabeza?): {e}")
    print("Usando resolución por defecto 1920x1080.")
    WIDTH, HEIGHT = 1920, 1080

# --- 3. GENERAR LA "VERDAD ABSOLUTA" (Roja) a 30 FPS ---
print(f"Generando 'Ground Truth' a {TRUTH_FPS} FPS (basado en 'experimentos_webcam_60fps.py')...")

# --- Lógica de 'experimentos_webcam_60fps.py' ---
def calcular_velocidad_constante(PPI, DISTANCIA_PANTALLA_CM, VELOCIDAD_ANGULAR_GRADOS=30.0):
    grados_por_pixel = np.degrees(np.arctan((2.54 / PPI) / DISTANCIA_PANTALLA_CM))
    velocidad_pixeles_seg = VELOCIDAD_ANGULAR_GRADOS / grados_por_pixel
    return velocidad_pixeles_seg

PPI = 96.0
DISTANCIA_PANTALLA_CM = 60.0
k = 0.5              
theta_max = 15 * np.pi
velocity = calcular_velocidad_constante(PPI, DISTANCIA_PANTALLA_CM)
# --- Fin de los parámetros copiados ---

CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
max_r_x = (WIDTH / 2) * 0.95
max_r_y = (HEIGHT / 2) * 0.95
r_values_math = k * theta_max
A = max_r_x / r_values_math
B = max_r_y / r_values_math
num_points_math = 5000
theta_values = np.linspace(0, theta_max, num_points_math)
r_values = k * theta_values
x_values = CENTER_X + A * r_values * np.cos(theta_values)
y_values = CENTER_Y + B * r_values * np.sin(theta_values)
arc_length = np.zeros(len(x_values))
for i in range(1, len(x_values)):
    dx = x_values[i] - x_values[i - 1]; dy = y_values[i] - y_values[i - 1]
    arc_length[i] = arc_length[i - 1] + np.sqrt(dx**2 + dy**2)
total_length = arc_length[-1]
time_values = arc_length / total_length # Rango 0.0 a 1.0
total_time_math_s = total_length / velocity # Duración matemática total en segundos

# 1. Generar la trayectoria COMPLETA a 30FPS
num_frames_truth = int(total_time_math_s * TRUTH_FPS)
truth_timestamps = np.linspace(0, total_time_math_s * 1000, num_frames_truth)
interp_time_percent = truth_timestamps / (total_time_math_s * 1000.0)
interp_x = np.interp(interp_time_percent, time_values, x_values)
interp_y = np.interp(interp_time_percent, time_values, y_values)

# 2. Invertirla (como en el script del experimento)
interp_x = interp_x[::-1]
interp_y = interp_y[::-1]
ground_truth_path = np.stack((interp_x, interp_y), axis=1).astype(int)
print(f"Generada 'Truth' (Roja): {len(ground_truth_path)} fotogramas @ 30 FPS (Duración: {total_time_math_s:.2f}s)")


# --- 4. CARGAR Y MAPEAR DATOS DE MIRADA (Azul) ---
print(f"Cargando datos de la espiral desde: {SPIRAL_CSV_PATH}")
try:
    df_gaze = pd.read_csv(SPIRAL_CSV_PATH)
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo. {e}")
    sys.exit()

# Filtrar datos válidos
df_valid = df_gaze[df_gaze['valid_deteccion'] == True].copy()
if df_valid.empty:
    print("No se encontraron detecciones válidas en el CSV.")
    sys.exit()
    
# Extraer la trayectoria de la mirada (gaze_x, gaze_y)
gaze_raw_x = df_valid['gaze_x'].values
gaze_raw_y = df_valid['gaze_y'].values
print(f"Cargada Mirada (Azul): {len(gaze_raw_x)} fotogramas (CSV)")

# Aplicar mapeo lineal (Robusto)
truth_x_min = np.min(ground_truth_path[:, 0])
truth_x_max = np.max(ground_truth_path[:, 0])
truth_y_min = np.min(ground_truth_path[:, 1])
truth_y_max = np.max(ground_truth_path[:, 1])
gaze_x_min = np.quantile(gaze_raw_x, 0.01)
gaze_x_max = np.quantile(gaze_raw_x, 0.99)
gaze_y_min = np.quantile(gaze_raw_y, 0.01)
gaze_y_max = np.quantile(gaze_raw_y, 0.99)
gaze_calibrated_x = np.interp(gaze_raw_x, 
                              [gaze_x_min, gaze_x_max], 
                              [truth_x_max, truth_x_min])
gaze_calibrated_y = np.interp(gaze_raw_y, 
                              [gaze_y_min, gaze_y_max], 
                              [truth_y_max, truth_y_min])

# Crear la trayectoria de mirada final (no se necesita interpolación)
gaze_path = np.stack((gaze_calibrated_x, gaze_calibrated_y), axis=1).astype(int)

# --- 5. BUCLE DE ANIMACIÓN CON PYGAME (SINCRONIZADO POR ÍNDICE) ---
print(f"\nIniciando animación en {WIDTH}x{HEIGHT}...")
print(f"Bucle a {ANIMATION_FPS} FPS")
print("Presiona ESC o cierra la ventana para salir.")

pygame.display.set_caption("Visualización de Seguimiento Ocular")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.font.init()
font = pygame.font.SysFont(None, 30)

# Colores
COLOR_TARGET = (255, 0, 0)         # Rojo
COLOR_GAZE = (0, 100, 255)       # Azul
COLOR_TARGET_TRAIL = (130, 0, 0) # Rojo oscuro
COLOR_GAZE_TRAIL = (0, 0, 130)   # Azul oscuro
COLOR_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)

running = True
frame_index_anim = 0 # Índice de la animación (60 FPS)

while running:
    # 1. Manejar eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                
    # 2. Obtener posiciones (¡LÓGICA DE SINCRONIZACIÓN 2:1!)
    
    # Índice del punto Rojo (Datos a 30 FPS)
    # se actualiza cada 2 fotogramas de la animación (60 / 2 = 30 FPS)
    frame_index_truth = frame_index_anim // 2 
    
    # Índice del punto Azul (Datos a 30 FPS)
    # se actualiza cada 2 fotogramas de la animación (60 / 2 = 30 FPS)
    frame_index_gaze = frame_index_anim // 2 # <-- CAMBIO 2: Sincronizado con el rojo
    
    # Condición de salida
    if frame_index_truth >= len(ground_truth_path) or frame_index_gaze >= len(gaze_path):
        print("Fin de los datos (Truth o Gaze).")
        running = False
        break
        
    truth_pos = ground_truth_path[frame_index_truth]
    gaze_pos = gaze_path[frame_index_gaze]

    # 3. Dibujar en la pantalla
    screen.fill(COLOR_BG)
    
    # Dibujar los rastros (trayectorias completas)
    pygame.draw.lines(screen, COLOR_TARGET_TRAIL, False, ground_truth_path, 3)
    pygame.draw.lines(screen, COLOR_GAZE_TRAIL, False, gaze_path, 2)
    
    # Dibujar los puntos actuales (brillantes)
    pygame.draw.circle(screen, COLOR_TARGET, (int(truth_pos[0]), int(truth_pos[1])), 15)
    pygame.draw.circle(screen, COLOR_GAZE, (int(gaze_pos[0]), int(gaze_pos[1])), 10)

    # Dibujar texto de tiempo
    current_time_s = frame_index_anim / ANIMATION_FPS # El tiempo se basa en el reloj de 60 FPS
    time_text = font.render(f"Tiempo: {current_time_s:.2f}s", True, COLOR_TEXT)
    screen.blit(time_text, (10, 10))
    
    # 4. Actualizar pantalla
    pygame.display.flip()
    
    # 5. Siguiente fotograma (a 60 FPS)
    frame_index_anim += 1
    clock.tick(ANIMATION_FPS) # Se limita a 60 FPS

# --- 6. Salir ---
pygame.quit()
print("Animación finalizada.")
sys.exit()