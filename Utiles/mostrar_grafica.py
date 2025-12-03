import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
from scipy.signal import savgol_filter

# ==========================================
#        CONFIGURACIÓN DE USUARIO
# ==========================================
NAME = "Victor"
EXP_NUM = 1  # 0, 1, 2, 3

# Rutas
INPUT_DATA_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Data\Experimento_{EXP_NUM}\{NAME}_data"

# CARPETA DE SALIDA PARA WORD
OUTPUT_HD_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Analizar_Data\Graficas_HD_Tesis"
os.makedirs(OUTPUT_HD_DIR, exist_ok=True)

# Parámetros
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3

def configurar_estilo():
    # Usamos un contexto de 'paper' para que las letras sean grandes y legibles al reducir la imagen en Word
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.titleweight'] = 'bold'

# ==========================================
#      CÁLCULO CINEMÁTICO
# ==========================================

def procesar_cinematica(df):
    df['time_s'] = df['timestamp_ms'] / 1000.0
    dt = df['time_s'].values
    
    vectores = df[['gaze_x', 'gaze_y', 'gaze_z']].values
    v_curr = vectores[:-1]; v_next = vectores[1:]
    
    dot = np.sum(v_curr * v_next, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    
    ang_step = np.degrees(np.arccos(dot))
    ang_step = np.append(ang_step, 0)
    
    df['pos_angular_total'] = np.cumsum(ang_step)
    
    vel_raw = np.gradient(df['pos_angular_total'], dt)
    try: df['vel_angular'] = savgol_filter(vel_raw, SAVGOL_WINDOW, SAVGOL_POLY)
    except: df['vel_angular'] = vel_raw
        
    acc_raw = np.gradient(df['vel_angular'], dt)
    try: df['acel_angular'] = savgol_filter(acc_raw, SAVGOL_WINDOW, SAVGOL_POLY)
    except: df['acel_angular'] = acc_raw
        
    return df

# ==========================================
#      GUARDADO EN ALTA DEFINICIÓN
# ==========================================

def guardar_cuatro_graficas(df, filename_base):
    
    # Parámetros para guardar
    SAVE_PARAMS = {
        'dpi': 300,              # Alta resolución para impresión
        'bbox_inches': 'tight',  # Quita bordes blancos extra
        'format': 'png'          # Formato sin pérdidas
    }
    
    grid_params = dict(linestyle='--', color='black', alpha=0.6)

    # --- 1. VECTOR MIRADA ---
    plt.figure(figsize=(8, 6)) # 8x6 pulgadas es buen tamaño para media página en Word
    plt.plot(df['time_s'], df['gaze_x'], label='Componente X', color='blue', alpha=0.8)
    plt.plot(df['time_s'], df['gaze_y'], label='Componente Y', color='green', alpha=0.8)
    plt.plot(df['time_s'], df['gaze_z'], label='Componente Z', color='red', alpha=0.8)
    plt.title(f'Vector de Mirada Normalizado)')
    plt.ylabel('Valor del Vector')
    plt.xlabel('Tiempo (s)')
    plt.legend(loc='lower right', frameon=True, framealpha=1.0)
    plt.grid(True, **grid_params)
    
    save_name = os.path.join(OUTPUT_HD_DIR, f"{filename_base}_01_Vector.png")
    plt.savefig(save_name, **SAVE_PARAMS)
    plt.close()

    # --- 2. POSICIÓN ANGULAR ---
    plt.figure(figsize=(8, 6))
    plt.plot(df['time_s'], df['pos_angular_total'], color='#8e44ad', label='Recorrido Acumulado')
    plt.title(f'Posición Angular Total)')
    plt.ylabel('Grados Recorridos (°)')
    plt.xlabel('Tiempo (s)')
    plt.legend(loc='lower right', frameon=True, framealpha=1.0)
    plt.grid(True, **grid_params)
    
    save_name = os.path.join(OUTPUT_HD_DIR, f"{filename_base}_02_Posicion.png")
    plt.savefig(save_name, **SAVE_PARAMS)
    plt.close()

    # --- 3. VELOCIDAD ANGULAR ---
    plt.figure(figsize=(8, 6))
    plt.plot(df['time_s'], df['vel_angular'], color='#e67e22', label='Velocidad')
    plt.fill_between(df['time_s'], df['vel_angular'], color='#e67e22', alpha=0.1)
    plt.title(f'Velocidad Angular)')
    plt.ylabel('Velocidad (°/s)')
    plt.xlabel('Tiempo (s)')
    plt.legend(loc='upper right', frameon=True, framealpha=1.0)
    plt.grid(True, **grid_params)
    
    save_name = os.path.join(OUTPUT_HD_DIR, f"{filename_base}_03_Velocidad.png")
    plt.savefig(save_name, **SAVE_PARAMS)
    plt.close()

    # --- 4. ACELERACIÓN ANGULAR ---
    plt.figure(figsize=(8, 6))
    plt.plot(df['time_s'], df['acel_angular'], color='#c0392b', label='Aceleración')
    plt.axhline(0, color='black', linewidth=1.5)
    plt.title(f'Aceleración Angular)')
    plt.ylabel('Aceleración (°/s²)')
    plt.xlabel('Tiempo (s)')
    plt.legend(loc='upper right', frameon=True, framealpha=1.0)
    plt.grid(True, **grid_params)
    
    save_name = os.path.join(OUTPUT_HD_DIR, f"{filename_base}_04_Aceleracion.png")
    plt.savefig(save_name, **SAVE_PARAMS)
    plt.close()

    print(f"   ✅ Gráficas HD guardadas para: {filename_base}")

# ==========================================
#              MAIN
# ==========================================

def main():
    configurar_estilo()
    print("="*60)
    print(f"   GENERADOR GRÁFICAS HD PARA TESIS - {NAME}")
    print("="*60)
    print(f"📂 Guardando en: {OUTPUT_HD_DIR}\n")
    
    archivos = glob.glob(os.path.join(INPUT_DATA_DIR, "*_data.csv"))
    
    # Filtro estricto para "intento_1"
    archivos_intento_1 = [f for f in archivos if "_intento_1_" in os.path.basename(f)]
    
    if not archivos_intento_1:
        print("❌ No se encontraron archivos del 'intento_1'.")
        return

    for file_path in archivos_intento_1:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            if 'valid_deteccion' in df.columns:
                if df['valid_deteccion'].dtype == object:
                    df['valid_deteccion'] = df['valid_deteccion'].astype(str).map({'True': True, 'False': False})
                df = df[df['valid_deteccion'] == True].copy()
            
            if df.empty: continue

            df = procesar_cinematica(df)
            guardar_cuatro_graficas(df, filename.replace("_data.csv", ""))
            
        except Exception as e:
            print(f"   ❌ Error en {filename}: {e}")

    print("\n✅ Proceso completado. Ve a la carpeta Graficas_HD_Tesis.")

if __name__ == "__main__":
    main()