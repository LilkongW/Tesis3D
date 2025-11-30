import pandas as pd
import numpy as np
import os
import glob
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress

# ==========================================
#            CONFIGURACIÓN DE USUARIO
# ==========================================
NAME = "Victor"
EXP_NUM = 1  # 0 (Punto Fijo), 1 (9 Puntos), 3 (Espiral)

# ------------------------------------------
# Rutas de Directorios
# ------------------------------------------
INPUT_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Data\Experimento_{EXP_NUM}\{NAME}_data"
STIMULUS_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Videos\Experimento_{EXP_NUM}\{NAME}"

BASE_OUTPUT_DIR = rf"C:\Users\Victor\Documents\Tesis3D\Analizar_Data\Resultados\Experimento_{EXP_NUM}"
TARGET_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, NAME)

os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(TARGET_OUTPUT_DIR, f"{NAME}_Exp{EXP_NUM}_Biometrico_Completo.csv")

# ------------------------------------------
# Parámetros Globales
# ------------------------------------------
OFFSET_GLOBAL_MS = 0  
VENTANA_LATENCIA = [-0.2, 0.8] 

# Procesamiento
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3

# Umbrales Biométricos
UMBRAL_VEL_FIJACION = 100.0   
UMBRAL_ACEL_SACADA = 250.0
MIN_DUR_FIJACION_MS = 150.0 
MAX_DUR_SACADA_S = 0.150      
MIN_FRAMES_PARPADEO = 3 

# ==========================================
#      FUNCIÓN 1: CARGAR STIMULUS (LOG)
# ==========================================

def obtener_eventos_reales(stimulus_path, tipo_exp):
    """Lee el log del experimento y extrae tiempos de eventos."""
    try:
        df_stim = pd.read_csv(stimulus_path)
        
        # Exp Continuo (3) o Estático (0): Solo nos interesa el inicio
        if tipo_exp in [0, 3]:
            start_time = df_stim['relative_time_s'].iloc[0]
            return [start_time]

        # Exp Discreto (1, 2): Nos interesan los saltos
        df_stim['cambio'] = (df_stim['stimulus_x'] != df_stim['stimulus_x'].shift()) | \
                            (df_stim['stimulus_y'] != df_stim['stimulus_y'].shift())
        
        df_stim.loc[0, 'cambio'] = True
        eventos = df_stim[df_stim['cambio'] == True]
        tiempos_reales = eventos['relative_time_s'].values.tolist()
        
        return tiempos_reales

    except Exception as e:
        print(f"      ⚠️ No se pudo cargar log de estímulos: {e}")
        return []

# ==========================================
#      FUNCIONES DE PROCESAMIENTO
# ==========================================

def detectar_parpadeos_raw(df_raw):
    """Detecta parpadeos en la data cruda."""
    if df_raw['valid_deteccion'].dtype == object:
        df_raw['valid_deteccion'] = df_raw['valid_deteccion'].astype(str).map({'True': True, 'False': False, 'true': True, 'false': False})
    
    df = df_raw[df_raw['timestamp_ms'] >= OFFSET_GLOBAL_MS].copy()
    if df.empty: return []

    df['grupo'] = (df['valid_deteccion'] != df['valid_deteccion'].shift()).cumsum()
    duraciones_parpadeos = []
    
    for _, datos in df[df['valid_deteccion'] == False].groupby('grupo'):
        if len(datos) > MIN_FRAMES_PARPADEO:
            t_inicio = datos['timestamp_ms'].iloc[0]
            t_fin = datos['timestamp_ms'].iloc[-1]
            duraciones_parpadeos.append(t_fin - t_inicio)
            
    return duraciones_parpadeos

def calcular_cinematica(df):
    """Calcula Velocidad y Aceleración angular."""
    df['timestamp_ms_sincro'] = df['timestamp_ms'] - OFFSET_GLOBAL_MS
    df['time_s'] = df['timestamp_ms_sincro'] / 1000.0
    df = df[df['time_s'] >= 0].copy()
    
    dt = df['time_s'].values
    if len(dt) < 2: return df

    vectores = df[['gaze_x', 'gaze_y', 'gaze_z']].values
    v_curr = vectores[:-1]; v_next = vectores[1:]
    dot = np.sum(v_curr * v_next, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang_step = np.degrees(np.arccos(dot))
    ang_step = np.append(ang_step, 0)
    
    df['amplitud_step'] = ang_step
    df['pos_total'] = np.cumsum(ang_step)
    
    vel_raw = np.gradient(df['pos_total'], dt)
    try: df['velocidad'] = savgol_filter(vel_raw, SAVGOL_WINDOW, SAVGOL_POLY)
    except: df['velocidad'] = vel_raw
        
    acc_raw = np.gradient(df['velocidad'], dt)
    try: df['aceleracion'] = savgol_filter(acc_raw, SAVGOL_WINDOW, SAVGOL_POLY)
    except: df['aceleracion'] = acc_raw
    
    return df

def detectar_eventos_oculomotores(df):
    """Separa Fijaciones y Sacadas."""
    if df.empty: return [], pd.DataFrame(), pd.DataFrame()

    # --- 1. FIJACIONES ---
    df['es_fijacion'] = df['velocidad'] < UMBRAL_VEL_FIJACION
    df['grp_fij'] = (df['es_fijacion'] != df['es_fijacion'].shift()).cumsum()
    
    lista_fij_duracion = []
    lista_fij_latencia = []
    
    for _, d in df.groupby('grp_fij'):
        if d['es_fijacion'].iloc[0]:
            dur = (d['time_s'].iloc[-1] - d['time_s'].iloc[0]) * 1000
            if dur >= MIN_DUR_FIJACION_MS:
                lista_fij_duracion.append(dur)
                lista_fij_latencia.append({'inicio': d['time_s'].iloc[0]})
    
    df_fijaciones_latencia = pd.DataFrame(lista_fij_latencia)

    # --- 2. SACADAS ---
    acc = df['aceleracion'].values
    p_pos, _ = find_peaks(acc, height=UMBRAL_ACEL_SACADA, distance=5)
    p_neg, _ = find_peaks(-acc, height=UMBRAL_ACEL_SACADA, distance=5)
    time_vals = df['time_s'].values
    sacadas = []
    
    for idx_start in p_pos:
        t_start = time_vals[idx_start]
        cands = p_neg[p_neg > idx_start]
        if len(cands) > 0:
            idx_end = cands[0]
            t_end = time_vals[idx_end]
            if (t_end - t_start) <= MAX_DUR_SACADA_S:
                seg = df.iloc[idx_start:idx_end+1]
                amp = seg['pos_total'].iloc[-1] - seg['pos_total'].iloc[0]
                vel_pico = seg['velocidad'].max()
                acc_pico = seg['aceleracion'].abs().max() # Restaurado
                dur = (t_end - t_start) * 1000
                sacadas.append({
                    'amplitud': abs(amp),
                    'velocidad_pico': vel_pico,
                    'aceleracion_pico': acc_pico,
                    'duracion': dur
                })
                
    return lista_fij_duracion, df_fijaciones_latencia, pd.DataFrame(sacadas)

def calcular_latencia_dinamica(df_fijaciones, tiempos_reales_estimulos):
    """Calcula latencia cruzando datos con el Log."""
    if df_fijaciones.empty or not tiempos_reales_estimulos: return np.nan, np.nan
    
    latencias = []
    # Usar lista completa si es 1 evento, o saltar el primero si son varios (arranque)
    lista_a_usar = tiempos_reales_estimulos if len(tiempos_reales_estimulos) == 1 else tiempos_reales_estimulos[1:]
    
    for t_target in lista_a_usar: 
        t_min = t_target + VENTANA_LATENCIA[0]
        t_max = t_target + VENTANA_LATENCIA[1]
        
        candidatas = df_fijaciones[(df_fijaciones['inicio'] >= t_min) & (df_fijaciones['inicio'] <= t_max)]
        
        if not candidatas.empty:
            latencia_ms = (candidatas.iloc[0]['inicio'] - t_target) * 1000
            if latencia_ms > 80: 
                latencias.append(latencia_ms)
                
    if not latencias: return np.nan, np.nan
    return np.mean(latencias), np.std(latencias)

# ==========================================
#      MÉTRICAS ESPECÍFICAS (EXP 0 y 3)
# ==========================================

def analizar_estabilidad_exp0(df_valid):
    """Estabilidad para Punto Fijo."""
    std_pos = df_valid['pos_total'].std()
    return std_pos

def analizar_pursuit_exp3(df_valid):
    """Seguimiento suave para Espiral."""
    mask_pursuit = (df_valid['velocidad'] > 5.0) & (df_valid['aceleracion'].abs() < UMBRAL_ACEL_SACADA)
    tiempo_pursuit = mask_pursuit.sum() * (df_valid['time_s'].iloc[1] - df_valid['time_s'].iloc[0])
    tiempo_total = df_valid['time_s'].iloc[-1] - df_valid['time_s'].iloc[0]
    
    ratio_pursuit = tiempo_pursuit / tiempo_total if tiempo_total > 0 else 0
    vel_media_pursuit = df_valid.loc[mask_pursuit, 'velocidad'].mean() if mask_pursuit.any() else 0
    
    return ratio_pursuit, vel_media_pursuit

# ==========================================
#      EXTRACCIÓN DE CARACTERÍSTICAS
# ==========================================

def extraer_features_intento(file_path):
    try:
        # Detectar tipo de experimento por nombre o usar global
        filename = os.path.basename(file_path)
        tipo_exp = EXP_NUM 
        
        if "punto_fijo" in filename: tipo_exp = 0
        elif "9_puntos" in filename: tipo_exp = 1
        elif "5_puntos" in filename: tipo_exp = 2
        elif "espiral" in filename: tipo_exp = 3

        # 1. Cargar Data
        df_raw = pd.read_csv(file_path)
        if len(df_raw) < 50: return None
        
        # 2. Cargar Log Estímulos
        nombre_base = filename.replace("_data.csv", "_stimulus.csv")
        stimulus_path = os.path.join(STIMULUS_DIR, nombre_base)
        
        tiempos_reales = []
        if os.path.exists(stimulus_path):
            tiempos_reales = obtener_eventos_reales(stimulus_path, tipo_exp)
            print(f"      ✓ Log cargado ({len(tiempos_reales)} eventos)")
        else:
            print(f"      ⚠️ Log no encontrado. Usando tiempos teóricos.")
            if tipo_exp in [0, 3]: tiempos_reales = [0.0]
            else: tiempos_reales = [i * 2 for i in range(9)]

        # 3. Parpadeos
        parpadeos_durs = detectar_parpadeos_raw(df_raw)
        
        # 4. Cinemática
        if df_raw['valid_deteccion'].dtype == object:
             df_raw['valid_deteccion'] = df_raw['valid_deteccion'].astype(str).map({'True': True, 'False': False, 'true': True, 'false': False})
             
        df_valid = df_raw[df_raw['valid_deteccion'] == True].sort_values('frame_number').copy()
        df_valid = calcular_cinematica(df_valid)
        if df_valid.empty: return None 
        
        fijaciones_durs, df_fij_lat, df_sac = detectar_eventos_oculomotores(df_valid)
        
        # --- VECTOR DE CARACTERÍSTICAS ---
        features = {'archivo': filename, 'tipo_exp': tipo_exp}
        
        # Métricas Globales
        tiempo_total_s = df_valid['time_s'].iloc[-1] - df_valid['time_s'].iloc[0]
        features['duracion_total_s'] = tiempo_total_s
        features['distancia_total_deg'] = df_valid['pos_total'].iloc[-1]
        features['tasa_parpadeo_hz'] = len(parpadeos_durs) / tiempo_total_s if tiempo_total_s > 0 else 0
        features['parpadeo_duracion_media_ms'] = np.mean(parpadeos_durs) if parpadeos_durs else 0
        
        # Fijaciones & Latencia
        features['num_fijaciones'] = len(fijaciones_durs)
        features['fijacion_duracion_media_ms'] = np.mean(fijaciones_durs) if fijaciones_durs else 0
        
        lat_media, lat_std = calcular_latencia_dinamica(df_fij_lat, tiempos_reales)
        features['latencia_promedio_ms'] = lat_media
        features['latencia_std_ms'] = lat_std
        
        # --- LÓGICA ESPECÍFICA ---
        if tipo_exp == 0: # PUNTO FIJO
            std_estabilidad = analizar_estabilidad_exp0(df_valid)
            features['estabilidad_std_deg'] = std_estabilidad
            features['score_calidad_fijacion'] = 1.0 / (std_estabilidad + 0.1) 
            
        elif tipo_exp == 3: # ESPIRAL
            ratio_pursuit, vel_pursuit = analizar_pursuit_exp3(df_valid)
            features['ratio_smooth_pursuit'] = ratio_pursuit
            features['velocidad_media_pursuit'] = vel_pursuit
            features['num_sacadas_correctivas'] = len(df_sac)
        
        else: # PUNTOS (SACADAS)
            features['num_sacadas'] = len(df_sac)
            
            if not df_sac.empty and len(df_sac) > 2:
                features['sacada_vel_pico_media'] = df_sac['velocidad_pico'].mean()
                features['sacada_amplitud_media'] = df_sac['amplitud'].mean() # Restaurado
                features['sacada_acel_pico_media'] = df_sac['aceleracion_pico'].mean() # Restaurado
                
                # A) MAIN SEQUENCE (Velocidad vs Amplitud)
                s_vel, i_vel, r_vel, _, _ = linregress(df_sac['amplitud'], df_sac['velocidad_pico'])
                features['main_seq_pendiente_k'] = s_vel
                features['main_seq_intercepto'] = i_vel
                features['main_seq_r2'] = r_vel**2
                
                # B) LINEAR SEQUENCE (Duración vs Amplitud) - NUEVO (Detector de Fatiga)
                s_dur, i_dur, r_dur, _, _ = linregress(df_sac['amplitud'], df_sac['duracion'])
                features['linear_seq_pendiente_m'] = s_dur # ms por grado
                features['linear_seq_intercepto'] = i_dur
                features['linear_seq_r2'] = r_dur**2
                
            else:
                features['main_seq_pendiente_k'] = 0
                features['linear_seq_pendiente_m'] = 0

        return features

    except Exception as e:
        print(f"❌ Error procesando {os.path.basename(file_path)}: {e}")
        return None

# ==========================================
#           EJECUCIÓN
# ==========================================

def main():
    print("="*60)
    print(f"   GENERADOR DE PERFIL BIOMÉTRICO (FULL) - {NAME}")
    print("="*60)
    print(f"📂 Inputs Data: {INPUT_DIR}")
    
    archivos = glob.glob(os.path.join(INPUT_DIR, "*_data.csv"))
    if not archivos:
        print("❌ No se encontraron archivos.")
        return

    dataset = []
    for f in archivos:
        if "Resumen" in os.path.basename(f): continue
        print(f"-> {os.path.basename(f)}")
        datos = extraer_features_intento(f)
        if datos: dataset.append(datos)
    
    if dataset:
        df_final = pd.DataFrame(dataset)
        
        # Columnas prioritarias dinámicas según el experimento encontrado
        cols_base = ['archivo', 'latencia_promedio_ms']
        
        # Si hay Main Sequence (Exp 1 o 2)
        if 'main_seq_pendiente_k' in df_final.columns:
            cols_base.extend(['main_seq_pendiente_k', 'linear_seq_pendiente_m', 'num_sacadas'])
            
        # Si hay Estabilidad (Exp 0)
        if 'estabilidad_std_deg' in df_final.columns:
            cols_base.append('estabilidad_std_deg')
            
        # Si hay Pursuit (Exp 3)
        if 'ratio_smooth_pursuit' in df_final.columns:
            cols_base.extend(['ratio_smooth_pursuit', 'velocidad_media_pursuit'])

        # Ordenar columnas para el CSV final
        cols_rest = [c for c in df_final.columns if c not in cols_base]
        df_final = df_final[cols_base + cols_rest]
        
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ REPORTE GENERADO: {OUTPUT_FILE}")
        print("-" * 60)
        # Mostrar solo columnas que existan y sean relevantes
        cols_to_show = [c for c in cols_base if c in df_final.columns]
        print(df_final[cols_to_show].head())
    else:
        print("\n⚠️ No se generaron datos válidos.")

if __name__ == "__main__":
    main()