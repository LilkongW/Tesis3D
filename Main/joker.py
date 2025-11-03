import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURACIÓN DE RUTAS ---

# --- ¡MODIFICA ESTAS 2 LÍNEAS! ---
# 1. Elige el archivo CSV de ENTRADA (el que tiene los datos de 'gaze_x, y, z')
INPUT_CSV_PATH = r"C:\Users\Victor\Documents\Tesis3D\Data\Experimento_1\Victoria_data\Victoria_intento_1_data.csv"

# 2. Define dónde se guardará la IMAGEN de SALIDA
OUTPUT_IMAGE_PATH = r"C:\Users\Victor\Documents\Tesis3D\Presentacion_imagenes\plot_vector_mirada.png"
# ----------------------------------

def main():
    print(f"Generando imagen de plot desde: {INPUT_CSV_PATH}")
    
    # --- 1. Cargar y Preparar Datos ---
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en: {INPUT_CSV_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        sys.exit(1)

    if df.empty:
        print("Error: El archivo CSV está vacío.")
        sys.exit(1)

    # Convertir tiempo a segundos
    df['timestamp_s'] = df['timestamp_ms'] / 1000.0
    
    # Poner NaN donde la detección no fue válida
    # Esto creará "huecos" en la gráfica, lo cual es correcto
    df.loc[df['valid_deteccion'] == False, ['gaze_x', 'gaze_y', 'gaze_z']] = np.nan
    
    # Extraer datos
    t_data = df['timestamp_s']
    x_data = df['gaze_x']
    y_data = df['gaze_y']
    z_data = df['gaze_z']

    # --- 2. Configurar y Dibujar la Gráfica ---
    print("Creando la gráfica...")
    
    # Crear la figura y los ejes
    # Aumentar el tamaño para que se vea mejor (ancho, alto) en pulgadas
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Dibujar las tres líneas
    ax.plot(t_data, x_data, lw=1.5, label='gaze_x (Horizontal)')
    ax.plot(t_data, y_data, lw=1.5, label='gaze_y (Vertical)')
    ax.plot(t_data, z_data, lw=1.5, label='gaze_z (Profundidad)', linestyle='--')
    
    # Configurar límites y etiquetas
    ax.set_xlim(t_data.min(), t_data.max())
    ax.set_ylim(-1.1, 1.1) # Rango de -1 a 1 para los vectores
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Componente del Vector de Mirada')
    
    # Añadir título y leyenda
    file_name = os.path.basename(INPUT_CSV_PATH)
    ax.set_title(f'Vector de Mirada 3D en el Tiempo\n(Archivo: {file_name})')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # Ajustar el layout para que no se corte nada
    fig.tight_layout()

    # --- 3. Guardar la Imagen ---
    try:
        # Guardar la figura en el archivo especificado
        plt.savefig(OUTPUT_IMAGE_PATH, dpi=300) # dpi=300 para alta resolución
        print(f"¡Imagen guardada exitosamente en: {OUTPUT_IMAGE_PATH}")
    except Exception as e:
        print(f"Error al guardar la imagen: {e}")
    finally:
        # Cerrar la figura para liberar memoria
        plt.close(fig)

if __name__ == "__main__":
    main()