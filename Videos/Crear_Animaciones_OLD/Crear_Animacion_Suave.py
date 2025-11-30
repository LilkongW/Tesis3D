import pygame
import numpy as np
import imageio
import os
import sys

# --- 2. OBTENER DIMENSIONES DE PANTALLA ---
print("Obteniendo resolución de pantalla con Pygame...")

pygame.init()
display_info = pygame.display.Info()
WIDTH, HEIGHT = display_info.current_w, display_info.current_h

# Creamos un Surface en memoria para dibujar
offscreen_surface = pygame.Surface((WIDTH, HEIGHT))


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
# frames = [] # 🔑 ELIMINADO: ¡No guardaremos fotogramas en RAM!

save_path = r"/home/vit/Documentos/Tesis3D/Videos"
os.makedirs(save_path, exist_ok=True)
output_path = os.path.join(save_path, "espiral_eliptica_animacion_sin_mostrar.mp4")

# 🔑 MODIFICACIÓN: Crear el "escritor" de video ANTES del bucle
writer = None # Inicializar en None
try:
    print(f"🎥 Preparando archivo de video en: {output_path}")
    writer = imageio.get_writer(output_path, fps=FPS)
except ValueError as e:
    if "Could not find a backend" in str(e):
        print("-" * 50)
        print("🚨 ERROR: FALLÓ LA INICIALIZACIÓN DEL VIDEO.")
        print("Debes instalar el backend FFMPEG. Ejecuta:")
        print("pip install 'imageio[ffmpeg]'")
        print("-" * 50)
    else:
         print(f"🚨 ERROR DESCONOCIDO AL INICIAR EL ESCRITOR: {e}")
    pygame.quit()
    sys.exit()
except Exception as e:
    print(f"🚨 OTRO ERROR al iniciar el escritor: {e}")
    pygame.quit()
    sys.exit()


# 9. Bucle de animación
try:
    print(f"Generando {num_frames} fotogramas en segundo plano...")
    while running:
        
        # 🔑 MODIFICACIÓN: Dibujar en el Surface en memoria
        offscreen_surface.fill(bg_color)

        # Dibujar el círculo en la posición actual
        current_pos = (int(interp_x[index]), int(interp_y[index]))
        pygame.draw.circle(offscreen_surface, circle_color, current_pos, circle_radius)

        # Capturar el fotograma del Surface en memoria
        frame = pygame.surfarray.array3d(offscreen_surface)

        # CORRECCIÓN: Transponer el arreglo para cambiar de (WIDTH, HEIGHT, 3) a (HEIGHT, WIDTH, 3)
        frame = np.swapaxes(frame, 0, 1)

        # 🔑 MODIFICACIÓN: Escribir el fotograma directamente al disco
        writer.append_data(frame)

        # Mover el círculo a la siguiente posición
        if index < len(interp_x) - 1:
            index += 1
            # Opcional: Mostrar progreso para bucles largos
            if index % (num_frames // 20) == 0: # Muestra el progreso más a menudo
                print(f"  Progreso: {index}/{num_frames} fotogramas ({int(index/num_frames*100)}%)")
        else:
            print("  Progreso: ¡Completado!")
            running = False

except Exception as e:
    print(f"Error durante la ejecución de Pygame: {e}")
    running = False

finally:
    pygame.quit() # Es buena práctica liberar los recursos de Pygame
    
    # 10. 🔑 MODIFICACIÓN: Cerrar el "escritor" de video
    if writer is not None:
        writer.close()
        print(f"✅ Video guardado como '{output_path}'")
    else:
        print("❌ No se generó ningún video (el 'writer' no se inicializó).")
    
    sys.exit()