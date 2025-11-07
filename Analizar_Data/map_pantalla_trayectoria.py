import pandas as pd
import numpy as np
import pygame 
import matplotlib.pyplot as plt
import math
import sys 

# --- 1. CONFIGURACIÓN ---
FPS = 60.0
OFFSET_MS = 500.0 # El mismo offset de tu 'analizador.py' para sincronizar

# ¡Ruta al CSV de la espiral que queremos analizar!
SPIRAL_CSV_PATH = "/home/vit/Documentos/Tesis3D/Data/Experimento_3/Victor_data/Victor_intento_1_data.csv"
# (Ya no se necesita la matriz de calibración)

# --- 2. GENERAR LA "VERDAD ABSOLUTA" (Ground Truth) ---
print("Generando trayectoria 'Ground Truth' de la espiral...")

# --- Usar las dimensiones reales de la pantalla ---
print("Obteniendo resolución de pantalla con Pygame...")
try:
    pygame.init()
    display_info = pygame.display.Info()
    WIDTH, HEIGHT = display_info.current_w, display_info.current_h
    pygame.quit()
    print(f"Resolución detectada: {WIDTH}x{HEIGHT}")
except Exception as e:
    print(f"Error al iniciar Pygame (¿entorno sin cabeza?): {e}")
    print("Usando resolución por defecto 1920x1080.")
    WIDTH, HEIGHT = 1920, 1080
# --- Fin de la corrección ---

CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
k = 1.0
theta_max = 15 * np.pi
velocity = 230
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

# --- 3. CARGAR DATOS CRUDOS DE LA ESPIRAL ---
print(f"Cargando datos de la espiral desde: {SPIRAL_CSV_PATH}")
try:
    df_gaze = pd.read_csv(SPIRAL_CSV_PATH)
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo. {e}")
    sys.exit()

# Filtrar datos válidos y aplicar offset
df_valid = df_gaze[df_gaze['valid_deteccion'] == True].copy()
if df_valid.empty:
    print("No se encontraron detecciones válidas en el CSV.")
    sys.exit()
    
df_valid['timestamp_ms'] = df_valid['timestamp_ms'] - OFFSET_MS
df_valid = df_valid[df_valid['timestamp_ms'] >= 0]

# Extraer la trayectoria de la mirada (gaze_x, gaze_y)
gaze_raw_x = df_valid['gaze_x'].values
gaze_raw_y = df_valid['gaze_y'].values

# --- 4. APLICAR MAPEO LINEAL (REGLA DE 3) ---
print("Aplicando mapeo lineal simple (escalar + mover + doble reflejar)...")

# Encontrar los rangos de la "Verdad Absoluta" (Píxeles)
truth_x_min = np.min(interp_x)
truth_x_max = np.max(interp_x)
truth_y_min = np.min(interp_y)
truth_y_max = np.max(interp_y)

# --- ¡CORRECCIÓN! Usar cuantiles para un mapeo robusto ---
# Encontrar los rangos de la Mirada Cruda (Vector)
# Usar 1% y 99% para ignorar outliers
gaze_x_min = np.quantile(gaze_raw_x, 0.01)
gaze_x_max = np.quantile(gaze_raw_x, 0.99)
gaze_y_min = np.quantile(gaze_raw_y, 0.01)
gaze_y_max = np.quantile(gaze_raw_y, 0.99)
# --- Fin de la corrección ---

print("\n--- Rangos de Mapeo Detectados (Robustos al 98%) ---")
print(f"  Gaze X:  [{gaze_x_min:.3f}, {gaze_x_max:.3f}]")
print(f"  Gaze Y:  [{gaze_y_min:.3f}, {gaze_y_max:.3f}]")
print(f"  Pixel X: [{truth_x_min:.0f}, {truth_x_max:.0f}]")
print(f"  Pixel Y: [{truth_y_min:.0f}, {truth_y_max:.0f}]")
print("Aplicando DOBLE REFLEXIÓN (X e Y)...")

# Mapear Gaze X -> Pixel X (Reflexión Horizontal)
gaze_calibrated_x = np.interp(gaze_raw_x, 
                              [gaze_x_min, gaze_x_max], 
                              [truth_x_max, truth_x_min])

# Mapear Gaze Y -> Pixel Y (Reflexión Vertical)
gaze_calibrated_y = np.interp(gaze_raw_y, 
                              [gaze_y_min, gaze_y_max], 
                              [truth_y_max, truth_y_min])

df_valid['gaze_pixel_x'] = gaze_calibrated_x
df_valid['gaze_pixel_y'] = gaze_calibrated_y


# --- 5A. ¡NUEVO! MOSTRAR GRÁFICO 1 (SOLO MIRADA CALIBRADA) ---
print(f"\nGenerando Gráfico 1 (Solo Mirada Calibrada) en {WIDTH}x{HEIGHT}...")
fig_calib, ax_calib = plt.subplots(figsize=(15, 9))
ax_calib.set_facecolor('white')

# Dibujar la trayectoria de la "Mirada" (verde)
ax_calib.plot(df_valid['gaze_pixel_x'], df_valid['gaze_pixel_y'], color='green', 
        lw=1.5, alpha=0.7, label=f'Trayectoria Calibrada (Lineal) (N={len(df_valid)})')

# Marcar inicio y fin de la trayectoria de MIRADA
start_gaze_x = df_valid['gaze_pixel_x'].iloc[0]
start_gaze_y = df_valid['gaze_pixel_y'].iloc[0]
end_gaze_x = df_valid['gaze_pixel_x'].iloc[-1]
end_gaze_y = df_valid['gaze_pixel_y'].iloc[-1]

# Marcar INICIO
ax_calib.scatter([start_gaze_x], [start_gaze_y], 
           s=200, facecolors='green', edgecolors='k', zorder=10, 
           label='Inicio Mirada')
ax_calib.text(start_gaze_x, start_gaze_y - 15, 'INICIO', 
        color='green', ha='center', weight='bold')

# Marcar FIN
ax_calib.scatter([end_gaze_x], [end_gaze_y], 
           s=200, facecolors='red', edgecolors='k', zorder=10, 
           label='Fin Mirada')
ax_calib.text(end_gaze_x, end_gaze_y + 25, 'FIN', 
        color='red', ha='center', weight='bold')

ax_calib.set_xlim(0, WIDTH)
ax_calib.set_ylim(0, HEIGHT)
ax_calib.invert_yaxis() # Y=0 arriba
ax_calib.set_title("Gráfico 1: Trayectoria de Mirada Calibrada (Lineal)")
ax_calib.set_xlabel("Píxeles X")
ax_calib.set_ylabel("Píxeles Y")
ax_calib.legend()
ax_calib.grid(True, linestyle='--', alpha=0.5)
ax_calib.set_aspect('equal')

print("Mostrando Gráfico 1. Cierra esta ventana para continuar...")
plt.show()
# --- Fin de la Sección 5A ---


# --- 5B. SINCRONIZAR Y CALCULAR ERROR ---
# (Se renombra la sección 5 a 5B)
print("\nSincronizando y calculando error dinámico...")
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
    
    if error_px < (WIDTH / 4): # Filtro de error máximo
        errores_en_pixeles.append(error_px)

# --- 6. REPORTAR Y VISUALIZAR (CALIBRADO) ---
if not errores_en_pixeles:
    print("No se pudieron calcular errores. ¿Datos vacíos o mala sincronización?")
    sys.exit()

mean_error = np.mean(errores_en_pixeles)
std_dev_error = np.std(errores_en_pixeles)
print("\n--- REPORTE DE PRECISIÓN (Mapeo Lineal Doble Reflexión) ---")
print(f"Error Promedio de Seguimiento: {mean_error:.2f} píxeles")
print(f"Desviación (Jitter) Promedio: {std_dev_error:.2f} píxeles")
print("--------------------------------------------------")

# --- Confirmar dimensiones y marcar puntos ---
print(f"\nGenerando Gráfico 2 (Mirada vs Verdad Absoluta) en {WIDTH}x{HEIGHT}...")
fig, ax = plt.subplots(figsize=(15, 9))
ax.set_facecolor('white')

# Dibujar la trayectoria "Verdadera" (rojo)
ax.plot(interp_x, interp_y, color='red', lw=4, 
        label=f'Trayectoria Real (Target) (N={len(ground_truth_path)})')
        
# Dibujar la trayectoria de la "Mirada" (verde)
ax.plot(df_valid['gaze_pixel_x'], df_valid['gaze_pixel_y'], color='green', 
        lw=1.5, alpha=0.7, label=f'Trayectoria Calibrada (Lineal) (N={len(df_valid)})')

# Marcar inicio y fin de la trayectoria de MIRADA
# (Se repite para que aparezca en ambos gráficos)
start_gaze_x = df_valid['gaze_pixel_x'].iloc[0]
start_gaze_y = df_valid['gaze_pixel_y'].iloc[0]
end_gaze_x = df_valid['gaze_pixel_x'].iloc[-1]
end_gaze_y = df_valid['gaze_pixel_y'].iloc[-1]

# Marcar INICIO
ax.scatter([start_gaze_x], [start_gaze_y], 
           s=200, facecolors='green', edgecolors='k', zorder=10, 
           label='Inicio Mirada')
ax.text(start_gaze_x, start_gaze_y - 15, 'INICIO', 
        color='green', ha='center', weight='bold')

# Marcar FIN
ax.scatter([end_gaze_x], [end_gaze_y], 
           s=200, facecolors='red', edgecolors='k', zorder=10, 
           label='Fin Mirada')
ax.text(end_gaze_x, end_gaze_y + 25, 'FIN', 
        color='red', ha='center', weight='bold')

ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.invert_yaxis() # Y=0 arriba
ax.set_title(f"Gráfico 2: Validación (Error Promedio: {mean_error:.2f} px)")
ax.set_xlabel("Píxeles X")
ax.set_ylabel("Píxeles Y")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect('equal')

print("Mostrando Gráfico 2...")
plt.show()