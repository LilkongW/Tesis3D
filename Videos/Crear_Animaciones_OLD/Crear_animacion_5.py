import pygame
import numpy as np
import imageio
import os

save_path = r"/home/victor/Documentos/Tesis3D/Videos"
os.makedirs(save_path, exist_ok=True)  # Crear la carpeta si no existe

# Inicializar pygame
pygame.init()

# Configuración de pantalla
display_info = pygame.display.Info()
WIDTH, HEIGHT = display_info.current_w, display_info.current_h  # Resolución de pantalla completa
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Fijaciones en Pantalla Completa")

# Configuración del círculo
circle_radius = 30  # Tamaño del punto
circle_color = (255, 0, 0)  # Rojo
bg_color = (0, 0, 0)  # negro

# Parámetros de la cuadrícula (aún los usamos para calcular las posiciones)
ROWS, COLS = 3, 3
cell_width = WIDTH // COLS
cell_height = HEIGHT // ROWS

# Duración: cada fijación dura 2000 ms
fixation_duration = 2000  # en milisegundos

# Variables de animación
clock = pygame.time.Clock()
frames = []

# --- INICIO DE LA MODIFICACIÓN ---

# Generar las posiciones deseadas usando los cálculos de la cuadrícula 3x3
# (Usamos las celdas 0,0 0,2 2,0 2,2 y 1,1)

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


# Crear la secuencia
sequence = [
    pos_top_left,
    pos_top_right,
    pos_bottom_left,
    pos_bottom_right,
    pos_center
]

# Repetir la secuencia 2 veces
positions = sequence * 2

# --- FIN DE LA MODIFICACIÓN ---


# Iniciar animación
running = True

for pos in positions:
    start_time = pygame.time.get_ticks()
    
    # Bucle para mantener el punto fijo durante 'fixation_duration'
    while pygame.time.get_ticks() - start_time < fixation_duration:
        # Manejar eventos (como presionar ESC para salir)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # Añadido: Salir con ESC
                    running = False
                    break
        if not running:
            break

        screen.fill(bg_color)

        # Dibujar la cuadrícula
        for i in range(1, ROWS):
            pygame.draw.line(screen, (200, 200, 200), (0, i * cell_height), (WIDTH, i * cell_height), 2)
        for j in range(1, COLS):
            pygame.draw.line(screen, (200, 200, 200), (j * cell_width, 0), (j * cell_width, HEIGHT), 2)

        # Dibujar el círculo en la posición actual
        x, y = pos
        pygame.draw.circle(screen, circle_color, (x, y), circle_radius)

        pygame.display.flip()

        # Capturar el fotograma
        frame = pygame.surfarray.array3d(screen)
        # Ajustar la orientación del frame para imageio
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        frames.append(frame)

        clock.tick(30)  # Limitar a 30 FPS

    if not running:
        break

pygame.quit()

# Guardar el video solo si se generaron frames
if frames:
    video_filename = os.path.join(save_path, "fijaciones_5puntos_x2.mp4") # Nombre de video cambiado
    imageio.mimsave(video_filename, frames, fps=30)
    print(f"✅ Video guardado como '{video_filename}'")
else:
    print("No se generaron frames (animación interrumpida).")