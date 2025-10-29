import pygame
import numpy as np
import imageio
import os
import sys # Importar para salida limpia

# 1. Inicializar pygame
pygame.init()

# 2. Configuración de pantalla
WIDTH, HEIGHT = 1280, 720 # Tamaño de ventana fijo para mejor grabación (Horizontal)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Partícula en Espiral Elíptica con Velocidad Constante")

# 3. Configuración visual
circle_radius = 20
circle_color = (255, 0, 0)
bg_color = (0, 0, 0)
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

# 4. Parámetros de la Espiral Elíptica
k = 1.0
theta_max = 15 * np.pi
velocity = 230

max_r = k * theta_max
A = (WIDTH / 2) / max_r * 0.95 
B = (HEIGHT / 2) / max_r * 0.95 

# 5. Calcular la trayectoria elíptica
num_points = 5000
theta_values = np.linspace(0, theta_max, num_points)
r_values = k * theta_values

x_values = CENTER_X + A * r_values * np.cos(theta_values)
y_values = CENTER_Y + B * r_values * np.sin(theta_values)

# 6. Calcular la longitud de arco real para movimiento uniforme
arc_length = np.zeros(len(x_values))
for i in range(1, len(x_values)):
    dx = x_values[i] - x_values[i - 1]
    dy = y_values[i] - y_values[i - 1]
    arc_length[i] = arc_length[i - 1] + np.sqrt(dx**2 + dy**2)

total_length = arc_length[-1]
time_values = arc_length / total_length

# 7. Interpolar a la velocidad deseada
FPS = 60
total_time = total_length / velocity
num_frames = int(total_time * FPS)

interp_time = np.linspace(0, 1, num_frames)

interp_x = np.interp(interp_time, time_values, x_values)
interp_y = np.interp(interp_time, time_values, y_values)

interp_x = interp_x[::-1]
interp_y = interp_y[::-1]

# 8. Variables de animación y guardado
running = True
index = 0
clock = pygame.time.Clock()
frames = []

save_path = r"/home/vit/Documentos/Tesis3D/Videos"
os.makedirs(save_path, exist_ok=True)

# 9. Bucle de animación
try:
    while running:
        # Manejar eventos de salida
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        clock.tick(FPS)

        screen.fill(bg_color)

        # Dibujar el círculo en la posición actual
        current_pos = (int(interp_x[index]), int(interp_y[index]))
        pygame.draw.circle(screen, circle_color, current_pos, circle_radius)

        pygame.display.flip()

        # Capturar el fotograma y corregir la orientación para imageio/FFMPEG 
        # pygame.surfarray.array3d(screen) produce (WIDTH, HEIGHT, 3).
        # imageio/FFMPEG espera (HEIGHT, WIDTH, 3).
        frame = pygame.surfarray.array3d(screen)
        
        # 🔑 CORRECCIÓN: Transponer el arreglo para cambiar de (WIDTH, HEIGHT, 3) a (HEIGHT, WIDTH, 3)
        frame = np.swapaxes(frame, 0, 1) # Intercambia el eje 0 (WIDTH) por el eje 1 (HEIGHT)

        frames.append(frame)

        # Mover el círculo a la siguiente posición
        if index < len(interp_x) - 1:
            index += 1
        else:
            running = False

except Exception as e:
    print(f"Error durante la ejecución de Pygame: {e}")
    running = False

finally:
    pygame.quit()
    
    # 10. Guardar la animación como video
    if frames:
        output_path = os.path.join(save_path, "espiral_eliptica_animacion.mp4")
        
        try:
            print(f"🎬 Iniciando guardado de video... ({len(frames)} fotogramas)")
            imageio.mimsave(output_path, frames, fps=FPS)
            print(f"✅ Video guardado como '{output_path}'")
        except ValueError as e:
            if "Could not find a backend" in str(e):
                print("-" * 50)
                print("🚨 ERROR: FALLÓ EL GUARDADO DEL VIDEO.")
                print("Debes instalar el backend FFMPEG. Ejecuta:")
                print("pip install 'imageio[ffmpeg]'")
                print("-" * 50)
            else:
                 print(f"🚨 ERROR DESCONOCIDO AL GUARDAR EL VIDEO: {e}")
        except Exception as e:
            print(f"🚨 OTRO ERROR al guardar el video: {e}")
    else:
        print("❌ No se capturaron fotogramas para guardar.")
    
    sys.exit()