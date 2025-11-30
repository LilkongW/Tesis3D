import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress

# ==========================================
#        CONFIGURACIÓN DE USUARIO
# ==========================================
NAME = "Victor"
EXP_NUM = 1  # 0, 1, 2, 3

# ------------------------------------------
# Rutas
# ------------------------------------------
INPUT_DATA_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Data\Experimento_{EXP_NUM}\{NAME}_data"
INPUT_STIM_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Videos\Experimento_{EXP_NUM}\{NAME}"
OUTPUT_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Analizar_Data\Graficas\Experimento_{EXP_NUM}\{NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------
# Parámetros de Procesamiento (Para detectar sacadas)
# ------------------------------------------
OFFSET_GLOBAL_S = 0.0
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3
UMBRAL_ACEL_SACADA = 250.0  # Umbral para detectar inicio de sacada
MAX_DUR_SACADA_S = 0.150    # Duración máxima lógica de una sacada

def configurar_estilo():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['lines.linewidth'] = 2

# ==========================================
#      FUNCIONES DE CÁLCULO CINEMÁTICO
# ==========================================

def calcular_metricas_sacadas(df):
    """
    Calcula velocidad, aceleración y extrae eventos de sacadas individuales
    para poder graficar la Main Sequence y Duración.
    """
    # 1. Preparar tiempo y vectores
    df['time_s'] = (df['timestamp_ms'] / 1000.0)
    dt = df['time_s'].values
    
    vectores = df[['gaze_x', 'gaze_y', 'gaze_z']].values
    v_curr = vectores[:-1]; v_next = vectores[1:]
    
    # Producto punto para ángulo (Amplitud paso a paso)
    dot = np.sum(v_curr * v_next, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang_step = np.degrees(np.arccos(dot))
    ang_step = np.append(ang_step, 0)
    
    # 2. Derivadas (Velocidad y Aceleración)
    pos_total = np.cumsum(ang_step)
    
    # Velocidad
    vel_raw = np.gradient(pos_total, dt)
    try: velocity = savgol_filter(vel_raw, SAVGOL_WINDOW, SAVGOL_POLY)
    except: velocity = vel_raw
    
    # Aceleración
    acc_raw = np.gradient(velocity, dt)
    try: acceleration = savgol_filter(acc_raw, SAVGOL_WINDOW, SAVGOL_POLY)
    except: acceleration = acc_raw
    
    # 3. Detectar Sacadas (Picos de Aceleración)
    p_pos, _ = find_peaks(acceleration, height=UMBRAL_ACEL_SACADA, distance=5)
    p_neg, _ = find_peaks(-acceleration, height=UMBRAL_ACEL_SACADA, distance=5)
    
    sacadas_data = []
    
    for idx_start in p_pos:
        # Buscar el final de la sacada (pico negativo posterior)
        cands = p_neg[p_neg > idx_start]
        if len(cands) > 0:
            idx_end = cands[0]
            t_start = dt[idx_start]
            t_end = dt[idx_end]
            duracion = t_end - t_start
            
            if duracion <= MAX_DUR_SACADA_S:
                # Extraer métricas de este evento
                amplitud = pos_total[idx_end] - pos_total[idx_start]
                vel_pico = np.max(velocity[idx_start:idx_end+1])
                
                sacadas_data.append({
                    'Amplitud_deg': abs(amplitud),
                    'Velocidad_Pico_deg_s': vel_pico,
                    'Duracion_ms': duracion * 1000.0
                })
                
    return pd.DataFrame(sacadas_data), velocity

# ==========================================
#      FUNCIONES DE CARGA
# ==========================================

def cargar_datos_ojo(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'valid_deteccion' in df.columns:
            if df['valid_deteccion'].dtype == object:
                df['valid_deteccion'] = df['valid_deteccion'].astype(str).map({'True': True, 'False': False})
            df = df[df['valid_deteccion'] == True].copy()
        
        # Calcular cinemática y obtener sacadas
        df_sacadas, velocidad_array = calcular_metricas_sacadas(df)
        
        # Añadir velocidad al df principal para gráficas temporales
        df['velocidad'] = velocidad_array
        
        return df, df_sacadas
    except Exception as e:
        print(f"❌ Error procesando {os.path.basename(file_path)}: {e}")
        return None, None

def cargar_datos_estimulo(data_filename):
    try:
        stim_filename = os.path.basename(data_filename).replace("_data.csv", "_stimulus.csv")
        stim_path = os.path.join(INPUT_STIM_DIR, stim_filename)
        if not os.path.exists(stim_path): return None
        return pd.read_csv(stim_path)
    except: return None

# ==========================================
#      FUNCIONES DE GRAFICADO
# ==========================================

def graficar_biometria_sacadas(df_sacadas, filename_base):
    """Genera las gráficas de Main Sequence y Duración."""
    if df_sacadas is None or len(df_sacadas) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- GRÁFICA 1: MAIN SEQUENCE (Velocidad vs Amplitud) ---
    sns.scatterplot(data=df_sacadas, x='Amplitud_deg', y='Velocidad_Pico_deg_s', 
                    ax=axes[0], color='crimson', s=60, alpha=0.7, edgecolor='k')
    
    # Regresión lineal para Main Sequence
    slope, intercept, r_value, _, _ = linregress(df_sacadas['Amplitud_deg'], df_sacadas['Velocidad_Pico_deg_s'])
    x_range = np.linspace(df_sacadas['Amplitud_deg'].min(), df_sacadas['Amplitud_deg'].max(), 100)
    y_pred = slope * x_range + intercept
    
    axes[0].plot(x_range, y_pred, color='navy', linestyle='--', linewidth=2, 
                 label=f'Ajuste (k={slope:.1f})')
    
    axes[0].set_title('Main Sequence (Dinámica)\nVelocidad vs Amplitud')
    axes[0].set_xlabel('Amplitud (Grados)')
    axes[0].set_ylabel('Velocidad Pico (°/s)')
    axes[0].legend()

    # --- GRÁFICA 2: LINEAR SEQUENCE (Duración vs Amplitud) ---
    sns.scatterplot(data=df_sacadas, x='Amplitud_deg', y='Duracion_ms', 
                    ax=axes[1], color='orange', s=60, alpha=0.7, edgecolor='k')
    
    # Regresión lineal para Duración
    slope_d, intercept_d, r_d, _, _ = linregress(df_sacadas['Amplitud_deg'], df_sacadas['Duracion_ms'])
    y_pred_d = slope_d * x_range + intercept_d
    
    axes[1].plot(x_range, y_pred_d, color='darkgreen', linestyle='--', linewidth=2, 
                 label=f'Ajuste (m={slope_d:.1f} ms/°)')
    
    axes[1].set_title('Eficiencia Temporal\nDuración vs Amplitud')
    axes[1].set_xlabel('Amplitud (Grados)')
    axes[1].set_ylabel('Duración (ms)')
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"Biometria_{filename_base}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def graficar_recorrido_y_temporal(df_ojo, df_stim, filename_base):
    """Panel visual 2D y Temporal."""
    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Espacial 2D
    ax1 = fig.add_subplot(gs[:, 0])
    sns.kdeplot(data=df_ojo, x='gaze_x', y='gaze_y', fill=True, cmap="Blues", alpha=0.3, ax=ax1)
    ax1.plot(df_ojo['gaze_x'], df_ojo['gaze_y'], color='navy', alpha=0.4, linewidth=0.5)
    
    if df_stim is not None:
        ax1.plot(df_stim['stimulus_x'], df_stim['stimulus_y'], 'ro--', linewidth=2, label='Target')
        
    ax1.set_title(f"Recorrido Ocular: {filename_base}")
    ax1.invert_yaxis()
    ax1.axis('equal')
    ax1.legend()

    # 2. Temporal X
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_ojo['timestamp_ms']/1000, df_ojo['gaze_x'], label='Ojo X')
    if df_stim is not None:
        ax2.step(df_stim['relative_time_s'], df_stim['stimulus_x'], 'r--', where='post', label='Target X')
    ax2.set_title("Posición Horizontal (X)")
    
    # 3. Velocidad (Nuevo)
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.plot(df_ojo['timestamp_ms']/1000, df_ojo['velocidad'], color='purple', label='Velocidad')
    ax3.set_title("Perfil de Velocidad (°/s)")
    ax3.set_xlabel("Tiempo (s)")
    ax3.fill_between(df_ojo['timestamp_ms']/1000, df_ojo['velocidad'], color='purple', alpha=0.1)

    save_path = os.path.join(OUTPUT_DIR, f"Recorrido_{filename_base}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()

# ==========================================
#              MAIN
# ==========================================

def main():
    configurar_estilo()
    print("="*60)
    print(f"   GENERADOR GRÁFICO BIOMÉTRICO - {NAME}")
    print("="*60)
    
    archivos = glob.glob(os.path.join(INPUT_DATA_DIR, "*_data.csv"))
    
    if not archivos:
        print("❌ No data encontrada.")
        return

    print(f"-> Procesando {len(archivos)} archivos...")
    
    for file_path in archivos:
        filename = os.path.basename(file_path)
        filename_base = filename.replace("_data.csv", "")
        
        if "Resumen" in filename or "Latencias" in filename: continue
            
        print(f"\nGenerando para: {filename_base}...")
        
        # 1. Cargar y procesar
        df_ojo, df_sacadas = cargar_datos_ojo(file_path)
        df_stim = cargar_datos_estimulo(file_path)
        
        if df_ojo is None or df_ojo.empty: continue
            
        # 2. Generar Panel Espacial/Temporal
        graficar_recorrido_y_temporal(df_ojo, df_stim, filename_base)
        
        # 3. Generar Panel Biométrico (Main Sequence + Duracion)
        if df_sacadas is not None and not df_sacadas.empty:
            graficar_biometria_sacadas(df_sacadas, filename_base)
            print(f"   ✓ Biometría generada ({len(df_sacadas)} sacadas)")
        else:
            print("   ⚠️ No se detectaron suficientes sacadas para biometría.")

    print("\n✅ Proceso completado.")

if __name__ == "__main__":
    main()