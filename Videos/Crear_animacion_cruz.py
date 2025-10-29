import pygame
import numpy as np
import imageio
import os
import sys

# Ruta de guardado (asegúrate de que sea la correcta para ti)
save_path = r"/home/vit/Documentos/Tesis3D/Videos"
os.makedirs(save_path, exist_ok=True)  # Crear la carpeta si no existe

# Inicializar pygame
pygame.init()

# Configuración de pantalla
display_info = pygame.display.Info()
WIDTH, HEIGHT = display_info.current_w, display_info.current_h  # Resolución de pantalla completa
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Movimiento en Cruz - Ida y Vuelta") # Título cambiado

# Configuración del círculo
circle_radius = 30  # Tamaño del punto
circle_color = (255, 0, 0)  # Rojo
bg_color = (0, 0, 0)  # negro

# --- INICIO DE LA MODIFICACIÓN ---

# Parámetros de animación
FPS = 30  # Fotogramas por segundo
MOVE_DURATION_SEC = 4  # ¡MÁS RÁPIDO! Duración de CADA movimiento (antes 6s)
NUM_FRAMES_PER_MOVE = int(MOVE_DURATION_SEC * FPS) # Frames por cada fase (4s * 30fps = 120)
PADDING = 60  # Píxeles de margen con el borde de la pantalla

# Coordenadas clave
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
Y_START = PADDING
Y_END = HEIGHT - PADDING
X_START = PADDING
X_END = WIDTH - PADDING

# 1. Generar trayectoria Horizontal (Izquierda -> Derecha)
x_values_lr = np.linspace(X_START, X_END, NUM_FRAMES_PER_MOVE)
coords_lr = [(int(x), CENTER_Y) for x in x_values_lr]

# 2. Generar trayectoria Horizontal (Derecha -> Izquierda)
x_values_rl = np.linspace(X_END, X_START, NUM_FRAMES_PER_MOVE)
coords_rl = [(int(x), CENTER_Y) for x in x_values_rl]

# 3. Generar trayectoria Vertical (Arriba -> Abajo)
y_values_ud = np.linspace(Y_START, Y_END, NUM_FRAMES_PER_MOVE)
coords_ud = [(CENTER_X, int(y)) for y in y_values_ud]

# 4. Generar trayectoria Vertical (Abajo -> Arriba)
y_values_du = np.linspace(Y_END, Y_START, NUM_FRAMES_PER_MOVE)
coords_du = [(CENTER_X, int(y)) for y in y_values_du]

# 5. Combinar todas las posiciones en una sola lista
# El orden será: L->R, R->L, U->D, D->U
all_positions = coords_lr + coords_rl + coords_ud + coords_du

# --- FIN DE LA MODIFICACIÓN ---

# Variables de animación
running = True
index = 0
clock = pygame.time.Clock()
frames = []

# Bucle de animación (itera por 'all_positions')
try:
    while running:
        # Manejar eventos (como presionar ESC para salir)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        if not running:
            break

        screen.fill(bg_color)

        # Dibujar el círculo en la posición actual
        current_pos = all_positions[index]
        pygame.draw.circle(screen, circle_color, current_pos, circle_radius)

        pygame.display.flip()

        # Capturar el fotograma
        frame = pygame.surfarray.array3d(screen)
        # Ajustar la orientación del frame para imageio
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        frames.append(frame)

        # Mover al siguiente frame de la animación
        if index < len(all_positions) - 1:
            index += 1
        else:
            running = False # Terminar cuando la animación se complete

        clock.tick(FPS) # Limitar a los FPS definidos

except Exception as e:
    print(f"Error durante la animación: {e}")
finally:
    pygame.quit()

# Guardar el video solo si se generaron frames
if frames:
    # Nombre de video cambiado
    video_filename = os.path.join(save_path, "movimiento_cruz_ida_y_vuelta.mp4") 
    try:
        imageio.mimsave(video_filename, frames, fps=FPS)
        print(f"✅ Video guardado como '{video_filename}'")
    except Exception as e:
        print(f"🚨 Error al guardar el video: {e}")
        if "ffmpeg" in str(e).lower():
            print("Asegúrate de tener 'imageio[ffmpeg]' instalado: pip install 'imageio[ffmpeg]'")
else:
    print("No se generaron frames (animación interrumpida).")

sys.exit() # Salida limpia