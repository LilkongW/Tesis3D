import pandas as pd
import numpy as np
import os
import glob
import warnings
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import skew, kurtosis, entropy, linregress

# Ignorar warnings de division por cero en regresiones cortas
warnings.filterwarnings('ignore')

# Obtener la ruta base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# =============================================================================
#  CONFIGURACIÓN DEL SISTEMA
# =============================================================================

CONFIG = {
    'EXP_NUM': 1,
    'PATHS': {
        'BASE': BASE_DIR
    },
    'PARAMS': {
        'SAMPLING_RATE': 120,     
        'FOV_H': 60,              
        'SAVGOL_WINDOW': 21,
        'SAVGOL_POLY': 3,
        'TIME_OFFSET_S': 0.600,
        
        # --- ESTRATEGIA DE VENTANAS ---
        'WINDOW_SIZE_S': 2.5,
        'WINDOW_STRIDE_S': 1.0,
        
        # Umbrales
        'UMBRAL_VEL_FIJACION': 100.0,
        'UMBRAL_ACEL_SACADA': 200.0,
        'MIN_DUR_FIJACION_S': 0.100,
        'MIN_DUR_SACADA_S': 0.008,      
        'MAX_DUR_SACADA_S': 0.200,
        
        # Pupila
        'PUPIL_MIN_SIZE': 20.0,
        'PUPIL_MAX_SIZE': 200.0,
        
        # Avanzados
        'MICROSACCADE_PERCENTILE': 10,
        'HFD_KMAX': 5, # K-max para Higuchi Fractal Dimension (bajo para velocidad)
    }
}

CONFIG['EXP_DIR'] = os.path.join(CONFIG['PATHS']['BASE'], "Data", f"Experimento_{CONFIG['EXP_NUM']}")

class BiometricPipeline:
    def __init__(self, config):
        self.cfg = config
        
    # =========================================================================
    # 0. UTILIDADES MATEMÁTICAS AVANZADAS
    # =========================================================================
    
    def higuchi_fractal_dimension(self, x, kmax):
        """Calcula la Dimensión Fractal de Higuchi (Complejidad de la señal)"""
        if len(x) < kmax * 2: return 0.0
        
        L = []
        x = np.array(x)
        N = len(x)
        
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(x[m + i * k] - x[m + (i - 1) * k])
                Lmk = (Lmk * (N - 1) / (((N - m) / k) * k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk) + 1e-10))
            
        K = np.log(1.0 / np.arange(1, kmax + 1))
        slope, _, _, _, _ = linregress(K, L)
        return slope  # HFD approx
        
    # =========================================================================
    # 1. ANÁLISIS DE PUPILA (DINÁMICA + VELOCIDAD)
    # =========================================================================
    
    def calcular_diametro_pupila(self, df):
        if 'ellipse_width' in df.columns and 'ellipse_height' in df.columns:
            return (df['ellipse_width'] + df['ellipse_height']) / 2.0
        elif 'pupil_diameter' in df.columns:
            return df['pupil_diameter']
        elif 'contour_area' in df.columns:
            return 2 * np.sqrt(df['contour_area'] / np.pi)
        return pd.Series(np.nan, index=df.index)
    
    def analizar_pupila_completo(self, df):
        pupil_diameter = self.calcular_diametro_pupila(df)
        valid_mask = (pupil_diameter > self.cfg['PARAMS']['PUPIL_MIN_SIZE']) & \
                     (pupil_diameter < self.cfg['PARAMS']['PUPIL_MAX_SIZE'])
        pupil_clean = pupil_diameter[valid_mask].values
        
        if len(pupil_clean) < 20:
            return {k: 0 for k in ['Pupil_Mean', 'Pupil_Std', 'Pupil_Vel_Max']}
        
        # 🌟 NUEVO: Velocidad Pupilar (Dinámica del SNA)
        # Derivada del diámetro respecto al tiempo
        pupil_vel = np.gradient(pupil_clean) * self.cfg['PARAMS']['SAMPLING_RATE']
        pupil_vel_max = np.max(np.abs(pupil_vel))
        
        metrics = {
            'Pupil_Mean': np.mean(pupil_clean),
            'Pupil_Std': np.std(pupil_clean),
            'Pupil_CV': np.std(pupil_clean) / np.mean(pupil_clean) if np.mean(pupil_clean) > 0 else 0,
            'Pupil_Vel_Max': pupil_vel_max
        }
        return metrics

    # =========================================================================
    # 2. CONTROL MOTOR (JERK + FRACTAL)
    # =========================================================================

    def calcular_jerk_metricas(self, df):
        if 'aceleracion' not in df.columns: 
            return {'Jerk_Mean': 0, 'Jerk_Max': 0}
        
        dt = 1 / self.cfg['PARAMS']['SAMPLING_RATE']
        jerk = np.gradient(df['aceleracion'], dt)
        
        return {
            'Jerk_Mean': np.mean(np.abs(jerk)),
            'Jerk_Max': np.max(np.abs(jerk))
        }

    def calcular_metricas_avanzadas_ventana(self, df):
        if 'velocidad' not in df.columns or len(df) < 20:
            return {'Velocity_Transition_Smoothness': 0.0, 'Microsaccade_Rate': 0.0, 'Fractal_Dim': 0.0}
        
        velocities = df['velocidad'].values
        
        # 1. Suavidad (Smoothness)
        acc = np.diff(velocities)
        jerk = np.diff(acc) if len(acc) > 1 else np.array([0])
        smoothness = 1.0 / (1.0 + np.std(jerk) / (np.ptp(velocities) + 1e-6))
        
        # 2. Microsacadas
        threshold = np.percentile(velocities, self.cfg['PARAMS']['MICROSACCADE_PERCENTILE'])
        peaks, _ = find_peaks(velocities, height=threshold*1.5, distance=2)
        # Filtrar picos muy altos (sacadas reales)
        peaks = [p for p in peaks if velocities[p] < threshold * 5]
        micro_rate = len(peaks) / (len(df) / self.cfg['PARAMS']['SAMPLING_RATE'])
        
        # 3. 🌟 NUEVO: Dimensión Fractal (Complejidad Cognitiva)
        # Usamos la señal de posición combinada o velocidad
        hfd = self.higuchi_fractal_dimension(velocities, self.cfg['PARAMS']['HFD_KMAX'])
        
        return {
            'Velocity_Transition_Smoothness': smoothness,
            'Microsaccade_Rate': micro_rate,
            'Fractal_Dim': hfd
        }

    # =========================================================================
    # 4. EVENTOS Y BIOMECÁNICA (MAIN SEQUENCE SLOPE)
    # =========================================================================
    
    def detectar_sacadas_y_fijaciones(self, df):
        acc = df['aceleracion'].values
        peaks, _ = find_peaks(acc, height=self.cfg['PARAMS']['UMBRAL_ACEL_SACADA'], distance=5)
        
        sacadas = []
        fijaciones = []
        
        for p in peaks:
            end_search = np.where(acc[p:] < 0)[0]
            if len(end_search) > 0:
                end = p + end_search[0]
                seg = df.iloc[p:end+1]
                dur = seg['time_s'].iloc[-1] - seg['time_s'].iloc[0]
                
                if self.cfg['PARAMS']['MIN_DUR_SACADA_S'] <= dur <= self.cfg['PARAMS']['MAX_DUR_SACADA_S']:
                    dx = seg['gaze_x'].iloc[-1] - seg['gaze_x'].iloc[0]
                    dy = seg['gaze_y'].iloc[-1] - seg['gaze_y'].iloc[0]
                    amp = np.sqrt(dx**2 + dy**2)
                    
                    if amp > 0:
                        sacadas.append({
                            'amp': amp,
                            'peak_vel': seg['velocidad'].max(),
                            't_start': seg['time_s'].iloc[0],
                            't_end': seg['time_s'].iloc[-1]
                        })
        
        df_sac = pd.DataFrame(sacadas)
        
        # Fijaciones simplificadas
        if len(df_sac) > 1:
            df_sac = df_sac.sort_values('t_start')
            for i in range(len(df_sac)-1):
                sac_end = df_sac.iloc[i]['t_end']
                sac_next_start = df_sac.iloc[i+1]['t_start']
                if (sac_next_start - sac_end) > self.cfg['PARAMS']['MIN_DUR_FIJACION_S']:
                    fix_seg = df[(df['time_s'] >= sac_end) & (df['time_s'] < sac_next_start)]
                    if len(fix_seg) > 5:
                        fijaciones.append({'vel_mean': fix_seg['velocidad'].mean()})
        
        return df_sac, pd.DataFrame(fijaciones)
    
    def analizar_eventos(self, df_sac, df_fix, duration):
        metrics = {
            'Saccade_Rate': 0, 'Main_Seq_Slope': 0, 
            'Fixation_Vel_Mean': 0
        }
        
        # --- SACADAS & MAIN SEQUENCE SLOPE ---
        if not df_sac.empty:
            metrics['Saccade_Rate'] = len(df_sac) / duration
            
            # 🌟 NUEVO: Main Sequence Slope (Regresión Vpeak vs Amp)
            # Biomecánica: V = K * Amp^c -> aprox lineal para sacadas cortas
            if len(df_sac) >= 3:
                slope, intercept, _, _, _ = linregress(df_sac['amp'], df_sac['peak_vel'])
                metrics['Main_Seq_Slope'] = slope
            elif len(df_sac) > 0:
                # Fallback: Ratio promedio si hay pocas sacadas
                metrics['Main_Seq_Slope'] = (df_sac['peak_vel'] / df_sac['amp']).mean()
            
        # --- FIJACIONES ---
        if not df_fix.empty:
            metrics['Fixation_Vel_Mean'] = df_fix['vel_mean'].mean()
            
        return metrics

    # =========================================================================
    # 5. RUN
    # =========================================================================
    
    def procesar_datos_raw(self, df_raw):
        df = df_raw.copy()
        if 'valid_deteccion' in df.columns:
            df = df[df['valid_deteccion'].astype(str) == 'True'].copy()
        if df.empty: return None

        t0 = df['timestamp_ms'].iloc[0]
        df['time_s'] = ((df['timestamp_ms'] - t0) / 1000.0) - self.cfg['PARAMS']['TIME_OFFSET_S']
        df = df[df['time_s'] >= 0].copy()
        if len(df) < 50: return None
        return df
    
    def calcular_cinematica(self, df):
        vecs = df[['gaze_x', 'gaze_y', 'gaze_z']].values
        normas = np.linalg.norm(vecs, axis=1, keepdims=True); normas[normas==0]=1.0
        vecs = vecs / normas
        dots = np.clip(np.sum(vecs[:-1] * vecs[1:], axis=1), -1.0, 1.0)
        ang_steps = np.insert(np.degrees(np.arccos(dots)), 0, 0)
        df['pos_deg'] = np.cumsum(ang_steps)
        
        w, p = self.cfg['PARAMS']['SAVGOL_WINDOW'], self.cfg['PARAMS']['SAVGOL_POLY']
        if len(df) > w:
            df['velocidad'] = savgol_filter(np.gradient(df['pos_deg'], df['time_s']), w, p)
            df['aceleracion'] = savgol_filter(np.gradient(df['velocidad'], df['time_s']), w, p)
        else:
            df['velocidad'] = 0; df['aceleracion'] = 0
        return df

    def run(self):
        print("="*70)
        print("🧬 PIPELINE BIOMÉTRICO v6: JOYAS DE LA CORONA (Fractal + Slope + PupilVel)")
        print("="*70)
        
        participant_dirs = [d for d in os.listdir(self.cfg['EXP_DIR']) 
                           if os.path.isdir(os.path.join(self.cfg['EXP_DIR'], d)) and d.endswith('_data')]
        
        all_participant_data = []
        
        for participant_dir in participant_dirs:
            participant_name = participant_dir.replace('_data', '')
            participant_path = os.path.join(self.cfg['EXP_DIR'], participant_dir)
            output_dir = os.path.join(self.cfg['PATHS']['BASE'], "Analizar_Data", "Resultados", 
                                     f"Exp{self.cfg['EXP_NUM']}_{participant_name}")
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\n🔍 Procesando: {participant_name}")
            files = glob.glob(os.path.join(participant_path, "*_data.csv"))
            participant_dataset = []
            
            for f in files:
                if "Resumen" in f or "Features" in f: continue
                vid_id = os.path.basename(f).replace("_data.csv", "")
                try:
                    df_raw = pd.read_csv(f)
                    df_full = self.procesar_datos_raw(df_raw)
                    if df_full is None: continue

                    df_full = self.calcular_cinematica(df_full)
                    
                    max_time = df_full['time_s'].max()
                    current_time = 0
                    
                    while current_time + self.cfg['PARAMS']['WINDOW_SIZE_S'] <= max_time:
                        t_start = current_time
                        t_end = current_time + self.cfg['PARAMS']['WINDOW_SIZE_S']
                        df_win = df_full[(df_full['time_s'] >= t_start) & (df_full['time_s'] < t_end)].copy()
                        
                        if len(df_win) < (self.cfg['PARAMS']['WINDOW_SIZE_S'] * self.cfg['PARAMS']['SAMPLING_RATE'] * 0.5):
                            current_time += self.cfg['PARAMS']['WINDOW_STRIDE_S']
                            continue
                        
                        row = {
                            'Participant': participant_name,
                            'VideoID': vid_id,
                            'Window_Start': t_start
                        }
                        
                        # 1. Cinemática Básica
                        row['Vel_Mean'] = df_win['velocidad'].mean()
                        row['Acc_Max'] = df_win['aceleracion'].max()
                        row['Gaze_Z_Mean'] = df_win['gaze_z'].mean()
                        
                        # 2. Motor & Complejidad
                        row.update(self.calcular_jerk_metricas(df_win))
                        row.update(self.calcular_metricas_avanzadas_ventana(df_win)) # Incluye Fractal
                        
                        # 3. Pupila Avanzada
                        row.update(self.analizar_pupila_completo(df_win)) # Incluye Vel_Max
                        
                        # 4. Eventos Avanzados
                        df_sac, df_fix = self.detectar_sacadas_y_fijaciones(df_win)
                        row.update(self.analizar_eventos(df_sac, df_fix, self.cfg['PARAMS']['WINDOW_SIZE_S'])) # Incluye Slope
                        
                        participant_dataset.append(row)
                        current_time += self.cfg['PARAMS']['WINDOW_STRIDE_S']
                    
                except Exception as e:
                    print(f"   ⚠️ Error en {vid_id}: {str(e)}")
            
            if participant_dataset:
                df_part = pd.DataFrame(participant_dataset).fillna(0)
                out_path = os.path.join(output_dir, f"{participant_name}_BIOMETRIC_METRICS.csv")
                df_part.to_csv(out_path, index=False)
                print(f"   ✅ Guardado: {len(df_part)} ventanas.")
                all_participant_data.append(df_part)

        if all_participant_data:
            df_all = pd.concat(all_participant_data, ignore_index=True)
            cons_path = os.path.join(self.cfg['PATHS']['BASE'], "Analizar_Data", "Resultados", "Consolidado", "BIOMETRIC_METRICS_ALL.csv")
            os.makedirs(os.path.dirname(cons_path), exist_ok=True)
            df_all.to_csv(cons_path, index=False)
            
            print("\n✅ DATASET GENERADO CON ÉXITO")
            print("📊 Nuevas métricas integradas:")
            print("   • Main_Seq_Slope (Biomecánica)")
            print("   • Pupil_Vel_Max (SNA)")
            print("   • Fractal_Dim (Estrategia Cognitiva)")

if __name__ == "__main__":
    pipeline = BiometricPipeline(CONFIG)
    pipeline.run()