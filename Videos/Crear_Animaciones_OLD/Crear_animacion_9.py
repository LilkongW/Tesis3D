import pygame
import numpy as np
# import imageio # Ya no es necesario
import os

# Eliminamos las variables de guardado ya que solo vamos a reproducir
# save_path = r"/home/victor/Documentos/Tesis3D/Videos"
# os.makedirs(save_path, exist_ok=True) 

# Inicializar pygame
pygame.init()

# Configuración de pantalla
display_info = pygame.display.Info()
WIDTH, HEIGHT = display_info.current_w, display_info.current_h  # Resolución de pantalla completa
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Fijaciones en Cuadrícula 3x3")

# Configuración del círculo
circle_radius = 30  # Tamaño del punto
circle_color = (255, 0, 0)  # Rojo
bg_color = (0, 0, 0)  # Negro

# Parámetros de la cuadrícula
ROWS, COLS = 3, 3
cell_width = WIDTH // COLS
cell_height = HEIGHT // ROWS

# Duración: cada fijación dura 2000 ms
fixation_duration = 2000  # en milisegundos

# Variables de animación
clock = pygame.time.Clock()
# frames = [] # Ya no es necesario

# ----------------------------------------------------
# --- INICIO DE LA MODIFICACIÓN DE POSICIONES (9 PUNTOS) ---
# ----------------------------------------------------

# Función auxiliar para calcular el centro de una celda (fila, columna)
def get_cell_center(row, col):
    """Calcula las coordenadas (x, y) del centro de una celda (row, col) 
    en una cuadrícula de 3x3."""
    x = col * cell_width + cell_width // 2
    y = row * cell_height + cell_height // 2
    return (x, y)

# Generar las 9 posiciones deseadas en orden de lectura
positions = []
# Recorrer filas (0, 1, 2)
for r in range(ROWS):
    # Recorrer columnas (0, 1, 2)
    for c in range(COLS):
        # La posición se calcula para el centro de la celda (r, c)
        positions.append(get_cell_center(r, c))

# Opcional: repetir la secuencia (se mantiene como en el código original, 
# pero puedes eliminar '* 2' si solo quieres una pasada)
positions = positions * 2

# ----------------------------------------------------
# --- FIN DE LA MODIFICACIÓN DE POSICIONES ---
# ----------------------------------------------------


# Iniciar animación
running = True

# Bucle principal de la animación
for pos in positions:
    start_time = pygame.time.get_ticks()
    
    # Bucle para mantener el punto fijo durante 'fixation_duration'
    while pygame.time.get_ticks() - start_time < fixation_duration:
        
        # --- Manejar eventos ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # Salir con ESC
                    running = False
                    break
        if not running:
            break

        # --- Dibujo ---
        screen.fill(bg_color)

        # Dibujar la cuadrícula (como referencia visual)
        grid_color = (50, 50, 50) # Cambiado a un gris oscuro para ser menos invasivo
        for i in range(1, ROWS):
            pygame.draw.line(screen, grid_color, (0, i * cell_height), (WIDTH, i * cell_height), 2)
        for j in range(1, COLS):
            pygame.draw.line(screen, grid_color, (j * cell_width, 0), (j * cell_width, HEIGHT), 2)

        # Dibujar el círculo en la posición actual
        x, y = pos
        pygame.draw.circle(screen, circle_color, (x, y), circle_radius)

        # Actualizar la pantalla
        pygame.display.flip()

        # Limitar a 30 FPS para un consumo de CPU estable
        clock.tick(30) 

    if not running:
        break

# Limpieza y salida
pygame.quit()

print("🏁 Animación finalizada.")

# --- Se ha eliminado la lógica de guardado de video ---