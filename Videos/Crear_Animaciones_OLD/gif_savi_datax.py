import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import os

def generar_plot_posicion_velocidad_final(csv_path, video_id, window_length, polyorder, min_peak_distance_ms=150):
    """
    Genera dos subgráficas (Posición vs. Tiempo y Velocidad vs. Tiempo) 
    con los colores solicitados y las líneas verticales y puntos que indican 
    los picos de velocidad.
    """
    
    # --- 0. Configuración ---
    x_col_name = 'Timestamp_s'
    y_pos_name = 'Iris_Center_X_ROI'
    W = window_length
    P = polyorder
    peak_threshold = 20 # p/s
    
    # Etiquetas y Títulos
    titulo_figura = f"Análisis de Posición y Velocidad X (Video: {video_id})"
    titulo_pos = "Posición X vs Tiempo (Marcadores de Pico de Velocidad)"
    titulo_vel = "Velocidad X vs Tiempo"
    x_label = "Tiempo (s)"
    y_pos_label = "X (pixeles)"
    y_vel_label = r'$\frac{\Delta X}{\Delta t}$ (píxeles/s)'
    
    try:
        # 1. Cargar y Limpiar Datos
        df = pd.read_csv(csv_path)
        df[x_col_name] = pd.to_numeric(df[x_col_name], errors='coerce')
        df[y_pos_name] = pd.to_numeric(df[y_pos_name], errors='coerce')
        df.dropna(subset=[x_col_name, y_pos_name], inplace=True)
        df_filtered = df[df['Video'] == video_id].copy()
        
        tiempo = df_filtered[x_col_name].to_numpy() 
        posicion_x = df_filtered[y_pos_name].to_numpy() 
        
        # Verificar y ajustar filtro
        if W % 2 == 0: W += 1
        
        # 2. Suavizado de POSICIÓN y Cálculo de VELOCIDAD
        dt = np.mean(np.diff(tiempo))
        min_peak_distance_s = min_peak_distance_ms / 1000.0 
        min_distance_points = int(np.ceil(min_peak_distance_s / dt))
        
        # Suavizado de la posición (para plotear la línea de posición base)
        posicion_suavizada = savgol_filter(posicion_x, window_length=W, polyorder=P)

        # Cálculo de la Velocidad (Derivada)
        velocidad_x = savgol_filter(posicion_x, window_length=W, polyorder=P, deriv=1, delta=dt)
        
        # 3. Detección de Picos con Restricción de Distancia
        
        # Máximos (Velocidad > +20 p/s)
        peaks_pos_indices, _ = find_peaks(
            velocidad_x, 
            height=peak_threshold, 
            distance=min_distance_points 
        ) 
        
        # Mínimos (Velocidad < -20 p/s)
        peaks_neg_indices, _ = find_peaks(
            -velocidad_x, 
            height=peak_threshold,
            distance=min_distance_points
        )
        
        # 4. Preparación de la Figura con dos Subgráficas
        
        fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(titulo_figura, fontsize=18)
        
        # --- SUBGRÁFICA SUPERIOR: POSICIÓN (Línea Azul) ---
        
        ax_pos.set_title(titulo_pos, fontsize=14)
        ax_pos.set_ylabel(y_pos_label, fontsize=12, color='blue') # Etiqueta de posición en azul
        ax_pos.tick_params(axis='y', labelcolor='blue')
        
        # Ploteo de la Posición (suavizada)
        ax_pos.plot(tiempo, posicion_suavizada, 'b-', linewidth=2, alpha=0.8, label='Posición X Suavizada') # ⬅️ Línea Azul
        
        # --- SUBGRÁFICA INFERIOR: VELOCIDAD (Línea Verde) ---
        
        ax_vel.set_title(titulo_vel, fontsize=14)
        ax_vel.set_xlabel(x_label, fontsize=12)
        ax_vel.set_ylabel(y_vel_label, fontsize=14, color='green') # Etiqueta de velocidad en verde
        ax_vel.tick_params(axis='y', labelcolor='green')
        
        # Ploteo de la Velocidad (Línea principal)
        ax_vel.plot(tiempo, velocidad_x, 'g-', linewidth=2, alpha=0.8, label='Velocidad Suavizada') # ⬅️ Línea Verde
        
        # Líneas de umbral y cero
        ax_vel.axhline(peak_threshold, color='red', linestyle=':', linewidth=1)
        ax_vel.axhline(-peak_threshold, color='darkblue', linestyle=':', linewidth=1)
        ax_vel.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # 5. Iterar y Marcar Picos en AMBAS Gráficas
        
        all_peak_indices = np.concatenate([peaks_pos_indices, peaks_neg_indices])
        
        for idx in all_peak_indices:
            peak_time = tiempo[idx]
            peak_pos_value = posicion_suavizada[idx]
            peak_vel_value = velocidad_x[idx]
            
            # Determinar color: Rojo para velocidad positiva, Azul Oscuro para velocidad negativa
            color = 'red' if peak_vel_value > 0 else 'darkblue'
            
            # A. Marcar en Gráfica de VELOCIDAD (Punto que contrasta)
            ax_vel.plot(peak_time, peak_vel_value, 
                        'o', color=color, markersize=8, zorder=5) # ⬅️ Punto con color de contraste
            
            # B. Dibujar Línea Vertical Punteada en AMBAS Gráficas
            ax_pos.axvline(peak_time, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            ax_vel.axvline(peak_time, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            
            # C. Marcar el Punto en Gráfica de POSICIÓN (Punto que contrasta)
            ax_pos.plot(peak_time, peak_pos_value, 
                        'o', color=color, markersize=8, zorder=5) # ⬅️ Punto con color de contraste

        # 6. Configuraciones finales
        ax_pos.grid(True, linestyle=':', alpha=0.6)
        ax_vel.grid(True, linestyle=':', alpha=0.6)
        
        # Leyenda personalizada
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='blue', lw=2, label='Posición X Suavizada'),
            Line2D([0], [0], color='green', lw=2, label='Velocidad X Suavizada'),
            Line2D([0], [0], color='red', marker='o', linestyle='', markersize=8, label='Pico Positivo (> 20 p/s)'),
            Line2D([0], [0], color='darkblue', marker='o', linestyle='', markersize=8, label='Pico Negativo (< -20 p/s)')
        ]
        ax_pos.legend(handles=custom_lines, loc='upper right', fontsize=10)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        print("Análisis final de Posición y Velocidad generado con los ajustes de color y formato.")

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

# --- Ejecución del Programa ---
if __name__ == "__main__":
    csv_file_path = r"C:\Users\Victor\Documents\Tesis3D\Experimento1OLD\Data_Raw_Victor.csv"
    video_a_analizar = 'Data_Victor_1.avi' 
    
    # Parámetros del filtro SG
    ventana = 25
    polinomio = 3
    
    # Distancia mínima requerida entre picos
    distancia_ms = 150 
    
    # ⬅️ EJECUTAR: Generar el análisis completo
    generar_plot_posicion_velocidad_final(csv_file_path, video_a_analizar, ventana, polinomio, distancia_ms)