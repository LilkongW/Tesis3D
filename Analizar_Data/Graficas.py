import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter, find_peaks

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
SAMPLING_RATE = 120   
SAVGOL_WIN = 21       
SAVGOL_POLY = 3
UMB_VEL_FIJ = 80.0    
UMB_ACEL_SAC = 250.0  
TIME_OFFSET = 0.600   # <--- Desplazamiento del T=0 (600ms)

# Obtener la ruta base: subir un nivel desde el directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Esto sube un nivel a Tesis3D C:\Users\Victor\Documents\Tesis3D

# Rutas predefinidas (Modifica si cambian)
DATA_PATH = r"C:\Users\Victor\Documents\Tesis3D\Data\Experimento_1\Victor_data\Victor_9_puntos_intento_1_data.csv"
STIM_PATH = r"C:\Users\Victor\Documents\Tesis3D\Videos\Experimento_1\Victor\Victor_9_puntos_intento_1_stimulus.csv"

def cargar_estimulos(filepath):
    """Carga los tiempos del estímulo."""
    if not os.path.exists(filepath):
        print("⚠️ No se encontró archivo de estímulos.")
        return []
    
    df = pd.read_csv(filepath)
    # Detectar cambios
    df['change'] = (df['stimulus_x'] != df['stimulus_x'].shift()) | \
                   (df['stimulus_y'] != df['stimulus_y'].shift())
    
    # Tomamos los tiempos tal cual vienen (asumiendo que empiezan en 0 relative_time)
    eventos = df[df['change'] == True]['relative_time_s'].values
    print(f"🔹 Eventos de estímulo detectados: {len(eventos)}")
    return eventos

def cargar_y_procesar(filepath):
    print(f"📂 Cargando datos del ojo: {os.path.basename(filepath)}...")
    df = pd.read_csv(filepath)
    
    if 'valid_deteccion' in df.columns:
        df = df[df['valid_deteccion'].astype(str) == 'True'].copy()
    
    # 1. Calcular tiempo absoluto original
    raw_time = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0]) / 1000.0
    
    # 2. APLICAR OFFSET (Desplazar el cero, NO borrar datos)
    df['time_s'] = raw_time - TIME_OFFSET
    
    # 3. Calcular Cinemática Vectorial 3D
    vecs = df[['gaze_x', 'gaze_y', 'gaze_z']].values
    normas = np.linalg.norm(vecs, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    vecs = vecs / normas
    
    # Ángulo entre frames
    dots = np.clip(np.sum(vecs[:-1] * vecs[1:], axis=1), -1.0, 1.0)
    ang_step = np.insert(np.degrees(np.arccos(dots)), 0, 0)
    
    # Acumular recorrido
    df['pos_deg'] = np.cumsum(ang_step)
    
    # Derivadas
    vel = np.gradient(df['pos_deg'], df['time_s'])
    df['velocidad'] = savgol_filter(vel, SAVGOL_WIN, SAVGOL_POLY)
    
    acc = np.gradient(df['velocidad'], df['time_s'])
    df['aceleracion'] = savgol_filter(acc, SAVGOL_WIN, SAVGOL_POLY)
    
    return df

def detectar_eventos(df):
    if df.empty: return [], []
    
    # Solo detectamos eventos en la parte "real" del experimento (t >= 0)
    # para no contar ruido de calibración previo
    mask_exp = df['time_s'] >= 0
    df_exp = df[mask_exp]
    
    if df_exp.empty: return [], []

    # 1. Sácadas
    peaks, _ = find_peaks(df_exp['aceleracion'], height=UMB_ACEL_SAC, distance=5)
    saccades = []
    for p in peaks:
        # Ajustar índice relativo al dataframe original
        idx_real = df_exp.index[p]
        # Buscar limites en el df original
        idx_num = df.index.get_loc(idx_real)
        start = max(0, idx_num-5)
        end = min(len(df)-1, idx_num+10)
        saccades.append((df['time_s'].iloc[start], df['time_s'].iloc[end]))
        
    # 2. Fijaciones
    is_fix = df_exp['velocidad'] < UMB_VEL_FIJ
    groups = (is_fix != is_fix.shift()).cumsum()
    fixations = []
    
    for _, g in df_exp[is_fix].groupby(groups):
        dur = g['time_s'].iloc[-1] - g['time_s'].iloc[0]
        if dur > 0.05: 
            fixations.append((g['time_s'].iloc[0], g['time_s'].iloc[-1]))
            
    return saccades, fixations

def graficar_clasico(df, saccades, fixations, stimulus_times):
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    t = df['time_s']
    
    # Línea vertical en T=0 (Inicio real)
    for ax in [ax1, ax2, ax3]:
        ax.axvline(0, color='black', linewidth=2, linestyle='-', alpha=0.8, label='Inicio Exp (T=0)' if ax == ax1 else "")

    # --- GRÁFICO 1: RECORRIDO ---
    # Calcular recorrido solo durante el experimento para mostrar en título
    recorrido_exp = df[df['time_s']>=0]['pos_deg'].iloc[-1] - df[df['time_s']>=0]['pos_deg'].iloc[0]
    
    ax1.plot(t, df['pos_deg'], 'b-', linewidth=1.5, label='Recorrido Total')
    ax1.set_title(f"Recorrido Angular Total (t>0): {recorrido_exp:.1f}°", fontweight='bold')
    ax1.set_ylabel("Grados acumulados")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- GRÁFICO 2: VELOCIDAD ---
    ax2.plot(t, df['velocidad'], 'k-', linewidth=1, label='Velocidad Ojo')
    for s, e in fixations: ax2.axvspan(s, e, color='green', alpha=0.2)
    for s, e in saccades: ax2.axvspan(s, e, color='red', alpha=0.3)
    
    ax2.set_title(f"Velocidad (Sacadas: {len(saccades)} | Fijaciones: {len(fixations)})", fontweight='bold')
    ax2.set_ylabel("°/s")
    ax2.axhline(UMB_VEL_FIJ, color='g', linestyle='--', linewidth=1, label='Umbral Fijación')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # --- GRÁFICO 3: ACELERACIÓN ---
    ax3.plot(t, df['aceleracion'], color='#ff7f0e', linewidth=1, label='Aceleración')
    ax3.set_title("Aceleración Angular", fontweight='bold')
    ax3.set_ylabel("°/s²")
    ax3.set_xlabel("Tiempo (s) [0 = Inicio Estímulo]")
    ax3.grid(True, linestyle=':', alpha=0.6)

    # --- ESTIMULOS ---
    label_added = False
    for stim_t in stimulus_times:
        # Los estímulos ya están en tiempo relativo (0, 2, 4...), así que se alinean solos
        label = "Estímulo (Cambio)" if not label_added else ""
        for ax in [ax1, ax2, ax3]:
            ax.axvline(stim_t, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=label if ax == ax1 else "")
        label_added = True

    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    
    if os.path.exists(DATA_PATH):
        df = cargar_y_procesar(DATA_PATH)
        sac, fix = detectar_eventos(df)
        stim_times = cargar_estimulos(STIM_PATH)
        
        print(f"✅ Análisis completado con Offset de {TIME_OFFSET}s")
        
        # Análisis del segundo pico POSTERIOR al inicio (t > 0)
        df_exp = df[df['time_s'] > 0]
        peaks, _ = find_peaks(df_exp['velocidad'], height=UMB_VEL_FIJ)
        
        if len(peaks) >= 2:
            idx = peaks[1] # Segundo pico
            # Ajuste de índice para obtener valor real
            t_pico = df_exp['time_s'].iloc[idx]
            v_pico = df_exp['velocidad'].iloc[idx]
            print(f"\n🔎 SEGUNDO PICO (t > 0):")
            print(f"   • Tiempo: {t_pico:.3f}s")
            print(f"   • Valor:  {v_pico:.2f}°/s")
        
        graficar_clasico(df, sac, fix, stim_times)
    else:
        print(f"❌ No se encontró: {DATA_PATH}")