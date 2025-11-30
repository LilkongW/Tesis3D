import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from scipy.signal import savgol_filter

# ==========================================
#        CONFIGURACIÓN GLOBAL
# ==========================================
NAME = "Victor"
EXP_NUM = 1  # 1 (9 Puntos)

# 1. Directorio de Entrada (Data del Ojo)
INPUT_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Data\Experimento_{EXP_NUM}\{NAME}_data"

# 2. Directorio de Logs (Estímulos reales)
STIMULUS_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Videos\Experimento_{EXP_NUM}\{NAME}"

# 3. Directorio de Salida (Resultados)
# Aquí se guardarán el CSV consolidado y los gráficos
BASE_OUTPUT_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Analizar_Data\Resultados\Experimento_{EXP_NUM}"
TARGET_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, NAME)

# Crear carpeta de salida si no existe
os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)

# --- PARÁMETROS DE PROCESAMIENTO ---
OFFSET_GLOBAL_MS = 0          # Offset manual (0 si usamos logs sincronizados)
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3
UMBRAL_VEL_FIJACION = 100.0   # °/s (velocidad máxima para considerar fijación)
MIN_DUR_FIJACION_MS = 150.0   # ms (duración mínima para validar)

# Ventana de búsqueda de reacción: 
# Buscamos que el ojo se fije entre 0.1s y 1.0s DESPUÉS de que el punto se movió.
VENTANA_BUSQUEDA = [0.1, 1.0] 

def configurar_estilo():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['lines.linewidth'] = 1.5

# ==========================================
#      FUNCIONES AUXILIARES
# ==========================================

def obtener_tiempos_reales(stimulus_path):
    """Lee el log y devuelve los segundos exactos donde el punto cambió de posición."""
    try:
        df_stim = pd.read_csv(stimulus_path)
        # Detectar cambios en X o Y
        df_stim['cambio'] = (df_stim['stimulus_x'] != df_stim['stimulus_x'].shift()) | \
                            (df_stim['stimulus_y'] != df_stim['stimulus_y'].shift())
        df_stim.loc[0, 'cambio'] = True
        
        # Filtrar eventos
        eventos = df_stim[df_stim['cambio'] == True]
        tiempos = eventos['relative_time_s'].values.tolist()
        return tiempos
    except Exception as e:
        print(f"      ⚠️ Error leyendo log {os.path.basename(stimulus_path)}: {e}")
        return []

# ==========================================
#      MÓDULO DE PROCESAMIENTO (Individual)
# ==========================================

def procesar_intento(file_path):
    """Procesa un solo archivo y devuelve sus latencias usando el LOG REAL."""
    try:
        # 1. Identificar archivos
        filename = os.path.basename(file_path)
        stim_name = filename.replace("_data.csv", "_stimulus.csv")
        stim_path = os.path.join(STIMULUS_DIR, stim_name)
        
        # 2. Obtener Tiempos Reales del Estímulo
        if os.path.exists(stim_path):
            tiempos_objetivo = obtener_tiempos_reales(stim_path)
        else:
            print(f"      ⚠️ Log no encontrado para {filename}. Usando tiempos teóricos.")
            tiempos_objetivo = [i * 2 for i in range(9)] # Fallback
            
        # 3. Cargar Data del Ojo
        df = pd.read_csv(file_path)
        if 'valid_deteccion' in df.columns:
            # Normalizar booleano si viene como string
            if df['valid_deteccion'].dtype == object:
                df['valid_deteccion'] = df['valid_deteccion'].astype(str).map({'True': True, 'False': False, 'true': True, 'false': False})
            
            # Filtramos solo detecciones válidas
            df = df[df['valid_deteccion'] == True].sort_values('frame_number')
        
        if df.empty: return None

        # Ajuste de tiempo
        df['time_s'] = (df['timestamp_ms'] - OFFSET_GLOBAL_MS) / 1000.0
        dt = df['time_s'].values
        if len(dt) < 2: return None

        # 4. Calcular Velocidad Angular
        vectores = df[['gaze_x', 'gaze_y', 'gaze_z']].values
        v_curr = vectores[:-1]; v_next = vectores[1:]
        dot = np.clip(np.sum(v_curr * v_next, axis=1), -1.0, 1.0)
        ang_step = np.degrees(np.arccos(dot))
        ang_step = np.append(ang_step, 0)
        
        pos_total = np.cumsum(ang_step)
        vel_raw = np.gradient(pos_total, dt)
        
        try: df['velocidad'] = savgol_filter(vel_raw, SAVGOL_WINDOW, SAVGOL_POLY)
        except: df['velocidad'] = vel_raw
        
        # 5. Detectar Inicios de Fijación
        df['es_fijacion'] = df['velocidad'] < UMBRAL_VEL_FIJACION
        df['grp'] = (df['es_fijacion'] != df['es_fijacion'].shift()).cumsum()
        
        fijaciones_inicio = []
        for _, d in df.groupby('grp'):
            if d['es_fijacion'].iloc[0]:
                dur = (d['time_s'].iloc[-1] - d['time_s'].iloc[0]) * 1000
                if dur >= MIN_DUR_FIJACION_MS:
                    fijaciones_inicio.append(d['time_s'].iloc[0])
        
        if not fijaciones_inicio: return None
        
        fijaciones_arr = np.array(fijaciones_inicio)

        # 6. Calcular Latencias (Cruce Estímulo vs Ojo)
        latencias = []
        
        # Ignoramos el primer evento (t=0)
        start_idx = 1 if len(tiempos_objetivo) > 1 else 0
        
        for i, t_est in enumerate(tiempos_objetivo[start_idx:], start=start_idx):
            t_min = t_est + VENTANA_BUSQUEDA[0]
            t_max = t_est + VENTANA_BUSQUEDA[1]
            
            # Buscar la primera fijación que ocurra DENTRO de la ventana tras el estímulo
            candidatas = fijaciones_arr[(fijaciones_arr >= t_min) & (fijaciones_arr <= t_max)]
            
            if len(candidatas) > 0:
                t_reaccion = candidatas[0]
                lat_ms = (t_reaccion - t_est) * 1000.0
                
                latencias.append({
                    'Archivo': filename,
                    'Estimulo_N': i+1,
                    'T_Estimulo': t_est,
                    'T_Reaccion': t_reaccion,
                    'Latencia_ms': lat_ms
                })
        
        return pd.DataFrame(latencias)

    except Exception as e:
        print(f"❌ Error procesando {os.path.basename(file_path)}: {e}")
        return None

# ==========================================
#      VISUALIZACIÓN GLOBAL
# ==========================================

def graficar_resultados_globales(df_total):
    
    # FIGURA: Histograma de Tiempos de Reacción
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Histograma con curva KDE
    sns.histplot(data=df_total, x='Latencia_ms', kde=True, bins=15, color='#2ecc71', alpha=0.6, ax=ax1)
    
    # Líneas de referencia
    media = df_total['Latencia_ms'].mean()
    mediana = df_total['Latencia_ms'].median()
    
    ax1.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.0f} ms')
    ax1.axvline(mediana, color='blue', linestyle='-', linewidth=2, label=f'Mediana: {mediana:.0f} ms')
    
    ax1.set_title(f'Distribución de Tiempos de Reacción - {NAME}', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Tiempo de Reacción (ms) [Estímulo -> Fijación]')
    ax1.set_ylabel('Frecuencia')
    ax1.legend()
    
    # Añadir caja de texto con stats
    textstr = '\n'.join((
        f'N Eventos: {len(df_total)}',
        f'Media: {media:.1f}ms',
        f'Std Dev: {df_total["Latencia_ms"].std():.1f}ms'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.show()

# ==========================================
#              MAIN
# ==========================================

def main():
    print("="*60)
    print(f"   ANÁLISIS DE TIEMPOS DE REACCIÓN - {NAME}")
    print("="*60)
    print(f"📂 Data Input: {INPUT_DIR}")
    print(f"📂 Logs Input: {STIMULUS_DIR}")
    print(f"📂 Resultados: {TARGET_OUTPUT_DIR}")
    
    archivos = glob.glob(os.path.join(INPUT_DIR, "*_data.csv"))
    
    if not archivos:
        print("❌ No se encontraron archivos de data.")
        return

    print(f"-> Procesando {len(archivos)} intentos...")
    
    todos_los_datos = []
    
    for f in archivos:
        if "Resumen" in f or "Latencias" in f: continue
        print(f"   Analizando: {os.path.basename(f)}...")
        df_lat = procesar_intento(f)
        if df_lat is not None and not df_lat.empty:
            todos_los_datos.append(df_lat)
    
    if not todos_los_datos:
        print("⚠️ No se pudieron extraer latencias válidas (Revisa umbrales o logs).")
        return

    # Unir resultados
    df_global = pd.concat(todos_los_datos, ignore_index=True)
    
    # --- REPORTE EN CONSOLA ---
    print("\n" + "="*40)
    print("      REPORTE GLOBAL DE REACCIÓN")
    print("="*40)
    print(f"Participante: {NAME}")
    print(f"Total eventos (sacadas válidas): {len(df_global)}")
    
    lat = df_global['Latencia_ms']
    
    print(f"\n--- ESTADÍSTICAS ---")
    print(f"• Promedio (Mean):    {lat.mean():.2f} ms")
    print(f"• Mediana (Median):   {lat.median():.2f} ms")
    print(f"• Desviación (Std):   {lat.std():.2f} ms")
    print(f"• Mejor tiempo (Min): {lat.min():.2f} ms")
    print(f"• Peor tiempo (Max):  {lat.max():.2f} ms")
    
    # Guardar CSV consolidado
    nombre_csv = f"{NAME}_Latencias_Detalladas.csv"
    ruta_salida = os.path.join(TARGET_OUTPUT_DIR, nombre_csv)
    
    df_global.to_csv(ruta_salida, index=False)
    print(f"\n✅ Archivo detallado guardado en:\n{ruta_salida}")
    
    # Graficar
    configurar_estilo()
    graficar_resultados_globales(df_global)

if __name__ == "__main__":
    main()