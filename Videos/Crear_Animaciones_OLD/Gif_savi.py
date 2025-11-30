import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import imageio
import imageio.v3 as iio
import os

# --- 1. Generación de la Señal de Prueba ---
np.random.seed(42)

# Parámetros
n_puntos = 1000
x = np.linspace(0, 10 * np.pi, n_puntos)

y_original = 2 * np.sin(x / 2) + 0.5 * np.cos(x * 1.5)
y_ruido = y_original + np.random.normal(0, 0.4, n_puntos)

# --- 2. Parámetros del Filtro Savitzky-Golay ---
window_length = 61   # Ventana grande
polyorder = 2        # Orden del polinomio

if window_length % 2 == 0:
    window_length += 1
if window_length <= polyorder:
    window_length = polyorder + 2

half_window = window_length // 2

# --- 3. Cálculo del Suavizado FINAL y Estable ---
y_suavizado_final = savgol_filter(y_ruido, window_length=window_length, polyorder=polyorder)

# --- 4. Preparación para la Generación del GIF ---
filenames = []
step_frames = 5

start_idx_processing = half_window 
end_idx_processing = n_puntos - half_window 

print(f"Generando frames: Ajuste Polinomial Local Extendido (Ventana={window_length}, Polinomio={polyorder})...")

for i in range(0, n_puntos, step_frames):
    
    plt.figure(figsize=(14, 8))
    
    # --- Elementos de la Gráfica ---
    plt.plot(x, y_ruido, 'b-', alpha=0.4, linewidth=1.5, label='Señal Ruidosa (Fondo)', zorder=1) 
    
    if i >= start_idx_processing:
        draw_until_idx = min(i, n_puntos)
        plt.plot(x[start_idx_processing:draw_until_idx], 
                 y_suavizado_final[start_idx_processing:draw_until_idx], 
                 'r-', linewidth=3.5, label='Señal Suavizada (Progreso)', zorder=3)
    
    if start_idx_processing <= i <= end_idx_processing:
        window_plot_start_idx = i - half_window
        window_plot_end_idx = min(i + half_window, n_puntos - 1) 
        
        # Extraer el buffer de datos ruidosos en la ventana actual
        buffer_x = x[window_plot_start_idx : window_plot_end_idx + 1]
        buffer_y_ruido = y_ruido[window_plot_start_idx : window_plot_end_idx + 1]

        # Ajustar el polinomio de grado 'polyorder'
        coeffs = np.polyfit(buffer_x, buffer_y_ruido, polyorder)
        polynomial = np.poly1d(coeffs)
        
        # ⬅️ CAMBIO CLAVE: Evaluar el polinomio sobre todo el rango 'x' de la señal
        y_fit_extended = polynomial(x) 
        
        # Graficar la curva polinómica ajustada extendida (línea verde clara)
        plt.plot(x, y_fit_extended, 'g--', alpha=0.6, linewidth=2.0, label=f'Polinomio Ajustado Localmente (Grado {polyorder})', zorder=4) # Línea completa
        
        # Sombreado de la ventana (para indicar dónde se hizo el ajuste)
        plt.axvspan(x[window_plot_start_idx], x[window_plot_end_idx], 
                    color='lime', alpha=0.2, label='Ventana de Datos para Ajuste')
        plt.axvline(x[i], color='darkgreen', linestyle='-', linewidth=2, 
                    label='Punto Central Suavizado')

    # --- Estilo y Títulos ---
    plt.title('Demostracion del Filtro Savitzky-Golay', fontsize=16)
    plt.xlabel('Tiempo / Eje X', fontsize=12)
    plt.ylabel('Amplitud / Eje Y', fontsize=12)
    
    # Ajuste dinámico de leyenda
    if i < start_idx_processing or i > end_idx_processing:
        legend_handles = [plt.Line2D([0], [0], color='blue', alpha=0.4, linewidth=1.5, label='Señal Ruidosa (Fondo)'),
                          plt.Line2D([0], [0], color='red', linewidth=3.5, label='Señal Suavizada (Progreso)')]
    else:
        legend_handles = [plt.Line2D([0], [0], color='blue', alpha=0.4, linewidth=1.5, label='Señal Ruidosa (Fondo)'),
                          plt.Line2D([0], [0], color='red', linewidth=3.5, label='Señal Suavizada (Progreso)'),
                          plt.Line2D([0], [0], color='green', linestyle='--', alpha=0.6, linewidth=2.0, label=f'Polinomio Ajustado Localmente (Grado {polyorder})'),
                          plt.Line2D([0], [0], color='darkgreen', linestyle='-', linewidth=2, label='Punto Central Suavizado')]

    plt.legend(handles=legend_handles, fontsize=10, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(min(y_ruido) - 1, max(y_ruido) + 1)
    
    filename = f'frame_{i:04d}.png'
    plt.savefig(filename, dpi=100)
    filenames.append(filename)
    plt.close()

# --- 5. Ensamblar el GIF ---
gif_filename = 'sg_polinomio_extendido_deslizante.gif'
print(f"\nEnsamblando el GIF: {gif_filename}...")

images = [iio.imread(filename) for filename in filenames]
imageio.mimsave(gif_filename, images, duration=50, loop=0) 

# --- 6. Limpieza de Archivos Temporales ---
print("Eliminando frames temporales...")
for filename in filenames:
    os.remove(filename)

print(f"\n✅ ¡Proceso finalizado! El GIF se ha guardado como: **{gif_filename}**")