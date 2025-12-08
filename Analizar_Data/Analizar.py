import pandas as pd
import numpy as np
import os
import glob
import traceback
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress, skew, kurtosis
from scipy.fft import fft, fftfreq

# Obtener la ruta base: subir un nivel desde el directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Esto sube un nivel a Tesis3D

# =============================================================================
#  CONFIGURACIÓN DEL SISTEMA
# =============================================================================

CONFIG = {
    'NAME': "Venegas",
    'EXP_NUM': 1,  # <--- 0=Fijo, 1=Lectura, 2=Saltos, 3=Espiral
    'PATHS': {
        'BASE': BASE_DIR
    },
    'PARAMS': {
        'SAMPLING_RATE': 120,     # Hz
        'FOV_H': 60,              # Grados
        'SAVGOL_WINDOW': 21,
        'SAVGOL_POLY': 3,
        'TIME_OFFSET_S': 0.600,   # Offset inicial para estabilizar
        
        # --- ESTRATEGIA DE VENTANAS (NUEVO) ---
        'WINDOW_SIZE_S': 2.5,     # Tamaño de la ventana (segundos)
        'WINDOW_STRIDE_S': 1.0,   # Paso de avance (1s = mucho solapamiento = más data)
        'MIN_DATA_IN_WINDOW': 0.8,# % mínimo de datos válidos para aceptar la ventana
        
        # Umbrales Biométricos
        'UMBRAL_VEL_FIJACION': 100.0, 
        'UMBRAL_ACEL_SACADA': 250.0,
        'MIN_DUR_FIJACION_S': 0.100,    # Reducido ligeramente para ventanas
        'MIN_DUR_SACADA_S': 0.020,      
        'MAX_DUR_SACADA_S': 0.200,
        
        # Latencia
        'LATENCY_MIN_S': 0.100,       
        'LATENCY_MAX_S': 0.800        
    }
}

# Configuración de Rutas
CONFIG['INPUT_DIR'] = os.path.join(CONFIG['PATHS']['BASE'], "Data", f"Experimento_{CONFIG['EXP_NUM']}", f"{CONFIG['NAME']}_data")
CONFIG['STIM_DIR'] = os.path.join(CONFIG['PATHS']['BASE'], "Videos", f"Experimento_{CONFIG['EXP_NUM']}", CONFIG['NAME'])
CONFIG['OUTPUT_DIR'] = os.path.join(CONFIG['PATHS']['BASE'], "Analizar_Data", "Resultados", f"Exp{CONFIG['EXP_NUM']}_{CONFIG['NAME']}")

class BiometricPipeline:
    def __init__(self, config):
        self.cfg = config
        os.makedirs(self.cfg['OUTPUT_DIR'], exist_ok=True)

    def load_stimulus_log(self, video_filename):
        """Carga el log de estímulos para calcular latencias o error."""
        base_name = video_filename.replace(".mp4", "").replace(".avi", "").replace("_data.csv", "")
        pattern = os.path.join(self.cfg['STIM_DIR'], f"*{base_name}*stimulus.csv")
        files = glob.glob(pattern)
        if files:
            try: return pd.read_csv(files[0])
            except: pass
        return None

    # =========================================================================
    # 1. MATEMÁTICA AVANZADA (SIGNAL PROCESSING)
    # =========================================================================

    def calcular_entropia_aproximada(self, U, m=2, r=0.2):
        """Calcula ApEn (Complejidad de la señal)."""
        try:
            U = np.array(U)
            N = len(U)
            if N < 20: return 0
            r = r * np.std(U)
            
            def _phi(m):
                z = N - m + 1
                x = np.array([U[i:i+m] for i in range(z)])
                C = np.zeros(z)
                for i in range(z):
                    d = np.max(np.abs(x - x[i]), axis=1)
                    C[i] = np.sum(d <= r) / z
                return np.sum(np.log(C)) / z
            
            return abs(_phi(m) - _phi(m + 1))
        except: return 0

    def calcular_fft_features(self, serie):
        """Separación de frecuencias (Temblor vs Movimiento)."""
        N = len(serie)
        if N < 10: return {'Power_Low': 0, 'Power_High': 0, 'Ratio_Freq': 0}
        
        yf = fft(serie)
        xf = fftfreq(N, 1 / self.cfg['PARAMS']['SAMPLING_RATE'])
        power = np.abs(yf[:N//2])
        freqs = xf[:N//2]
        
        # Bandas de interés biométrico
        p_low = np.sum(power[(freqs > 0.1) & (freqs <= 4)])   # Movimiento voluntario
        p_high = np.sum(power[(freqs > 10) & (freqs <= 30)])  # Micro-temblor muscular
        
        return {
            'Potencia_Freq_Baja': p_low,
            'Potencia_Freq_Alta': p_high,
            'Ratio_Freq_Alta_Baja': p_high / (p_low + 1e-6)
        }
    
    def calcular_perfil_ruido(self, df):
        """Calcula el ruido ocular durante periodos de 'casi' quietud (Micro-sacadas/Drift)."""
        quiet_periods = df[df['velocidad'] < 80.0] 
        if len(quiet_periods) < 10:
            return {'Noise_Std': 0, 'Noise_Kurtosis': 0}
        
        vels = quiet_periods['velocidad'].values
        return {
            'Noise_Std': np.std(vels),
            'Noise_Kurtosis': kurtosis(vels)
        }

    def calcular_latencia_reaccion(self, df_sacadas, df_stim, t_start_win, t_end_win):
        """Calcula latencia solo para estímulos dentro de la ventana actual."""
        if df_sacadas.empty or df_stim is None: return 0
        if 'relative_time_s' not in df_stim.columns: return 0
        
        # Detectar cambios en estimulo
        df_stim['change'] = (df_stim['stimulus_x'] != df_stim['stimulus_x'].shift()) | \
                            (df_stim['stimulus_y'] != df_stim['stimulus_y'].shift())
        
        event_times = df_stim.loc[df_stim['change'], 'relative_time_s'].values
        event_times = event_times - self.cfg['PARAMS']['TIME_OFFSET_S']
        
        relevant_events = event_times[(event_times >= t_start_win - 0.5) & (event_times <= t_end_win)]
        
        if len(relevant_events) == 0: return 0
        
        latencias = []
        sac_starts = df_sacadas['t_start'].values
        
        for t_stim in relevant_events:
            min_t = t_stim + self.cfg['PARAMS']['LATENCY_MIN_S']
            max_t = t_stim + self.cfg['PARAMS']['LATENCY_MAX_S']
            matches = sac_starts[(sac_starts >= min_t) & (sac_starts <= max_t)]
            if len(matches) > 0:
                latencias.append(matches[0] - t_stim)
                
        return np.mean(latencias) if latencias else 0

    # =========================================================================
    # 🔥 NUEVO: ANÁLISIS PROFUNDO DE MAIN SEQUENCE
    # =========================================================================
    
    def analizar_main_sequence_completo(self, df_sac):
        """
        Extrae métricas avanzadas de la Main Sequence que son únicas por persona.
        
        CONCEPTO: La Main Sequence relaciona amplitud con velocidad pico.
        Pero cada persona tiene un "estilo" único de cómo ejecuta esta relación:
        - Algunos son más "explosivos" (alta velocidad relativa)
        - Otros son más "suaves" (menor dispersión)
        - La consistencia (R²) es muy personal
        """
        metrics = {}
        
        if len(df_sac) < 5:
            # No hay suficientes datos
            return {
                'MainSeq_Slope': 0,
                'MainSeq_Intercept': 0,
                'MainSeq_R2': 0,
                'MainSeq_Residual_Std': 0,
                'MainSeq_Efficiency': 0,
                'MainSeq_Consistency_Score': 0,
                'PeakVel_Per_Degree': 0,
                'Amp_Vel_Correlation': 0,
                'MainSeq_Deviation_Score': 0
            }
        
        amps = df_sac['amp'].values
        vels = df_sac['peak_vel'].values
        
        # 1. Regresión Lineal Clásica
        slope, intercept, r_value, p_value, std_err = linregress(amps, vels)
        r_squared = r_value ** 2
        
        # 2. Calcular residuos (desviaciones de la línea ideal)
        predicted_vels = slope * amps + intercept
        residuals = vels - predicted_vels
        residual_std = np.std(residuals)
        
        # 3. 🔥 MÉTRICA ÚNICA: "Eficiencia de Main Sequence"
        # Ratio entre velocidad real y esperada (normalizado)
        # Personas "eficientes" tienen velocidades consistentemente altas
        mean_efficiency = np.mean(vels / (predicted_vels + 1e-6))
        
        # 4. 🔥 MÉTRICA ÚNICA: "Consistency Score"
        # Combina R² con la variabilidad de residuos
        # Alto = Muy predecible, Bajo = Errático
        consistency_score = r_squared * (1.0 / (residual_std + 1.0))
        
        # 5. 🔥 MÉTRICA ÚNICA: "Peak Velocity per Degree"
        # Velocidad promedio normalizada por amplitud
        # Captura la "agresividad" del movimiento
        peak_vel_per_deg = np.mean(vels / (amps + 0.1))  # +0.1 para evitar división por 0
        
        # 6. Correlación directa (alternativa a regresión)
        amp_vel_corr = np.corrcoef(amps, vels)[0, 1]
        
        # 7. 🔥 MÉTRICA ÚNICA: "Deviation Score"
        # Suma de desviaciones cuadráticas normalizadas
        # Penaliza outliers (personas inconsistentes)
        deviation_score = np.sqrt(np.mean(residuals**2)) / (np.mean(vels) + 1e-6)
        
        # 8. Análisis por rangos de amplitud (personas tienen preferencias)
        small_sac = df_sac[df_sac['amp'] < 5.0]  # Sacadas pequeñas
        large_sac = df_sac[df_sac['amp'] >= 5.0]  # Sacadas grandes
        
        if len(small_sac) > 0:
            metrics['Small_Saccade_Avg_Vel'] = small_sac['peak_vel'].mean()
        else:
            metrics['Small_Saccade_Avg_Vel'] = 0
            
        if len(large_sac) > 0:
            metrics['Large_Saccade_Avg_Vel'] = large_sac['peak_vel'].mean()
        else:
            metrics['Large_Saccade_Avg_Vel'] = 0
        
        # 9. 🔥 RATIO Small/Large (preferencia de velocidad por tamaño)
        if metrics['Large_Saccade_Avg_Vel'] > 0:
            metrics['Small_Large_Vel_Ratio'] = metrics['Small_Saccade_Avg_Vel'] / metrics['Large_Saccade_Avg_Vel']
        else:
            metrics['Small_Large_Vel_Ratio'] = 0
        
        # Guardar métricas principales
        metrics.update({
            'MainSeq_Slope': slope,
            'MainSeq_Intercept': intercept,
            'MainSeq_R2': r_squared,
            'MainSeq_Residual_Std': residual_std,
            'MainSeq_Efficiency': mean_efficiency,
            'MainSeq_Consistency_Score': consistency_score,
            'PeakVel_Per_Degree': peak_vel_per_deg,
            'Amp_Vel_Correlation': amp_vel_corr,
            'MainSeq_Deviation_Score': deviation_score
        })
        
        return metrics

    # =========================================================================
    # 🔥 NUEVAS MÉTRICAS DE DISTRIBUCIÓN DE SACADAS
    # =========================================================================
    
    def analizar_distribucion_sacadas(self, df_sac, t_start, t_end):
        """Extrae patrones de distribución temporal y espacial de sacadas."""
        metrics = {}
        
        if len(df_sac) == 0:
            return {
                'Saccade_Rate_Hz': 0,
                'ISI_Mean': 0,
                'ISI_Std': 0,
                'ISI_CV': 0,
                'Amp_Percentile_10': 0,
                'Amp_Percentile_90': 0,
                'Amp_Range': 0,
                'Amp_Std': 0,
                'PeakVel_Percentile_10': 0,
                'PeakVel_Percentile_90': 0,
                'PeakVel_Range': 0,
                'PeakVel_CV': 0
            }
        
        duration = t_end - t_start
        
        # 1. Tasa de sacadas (fundamental)
        metrics['Saccade_Rate_Hz'] = len(df_sac) / duration
        
        # 2. Inter-Saccadic Interval (ISI)
        if len(df_sac) > 1:
            t_ends = df_sac['t_start'].values + df_sac['dur'].values
            isis = df_sac['t_start'].values[1:] - t_ends[:-1]
            isis = isis[isis > 0]
            
            if len(isis) > 0:
                metrics['ISI_Mean'] = np.mean(isis)
                metrics['ISI_Std'] = np.std(isis)
                metrics['ISI_CV'] = np.std(isis) / (np.mean(isis) + 1e-6)  # Coef. variación
            else:
                metrics['ISI_Mean'] = 0
                metrics['ISI_Std'] = 0
                metrics['ISI_CV'] = 0
        else:
            metrics['ISI_Mean'] = 0
            metrics['ISI_Std'] = 0
            metrics['ISI_CV'] = 0
        
        # 3. Distribución de amplitudes
        amps = df_sac['amp'].values
        metrics['Amp_Percentile_10'] = np.percentile(amps, 10)
        metrics['Amp_Percentile_90'] = np.percentile(amps, 90)
        metrics['Amp_Range'] = metrics['Amp_Percentile_90'] - metrics['Amp_Percentile_10']
        metrics['Amp_Std'] = np.std(amps)
        
        # 4. Distribución de velocidades pico
        vels = df_sac['peak_vel'].values
        metrics['PeakVel_Percentile_10'] = np.percentile(vels, 10)
        metrics['PeakVel_Percentile_90'] = np.percentile(vels, 90)
        metrics['PeakVel_Range'] = metrics['PeakVel_Percentile_90'] - metrics['PeakVel_Percentile_10']
        metrics['PeakVel_CV'] = np.std(vels) / (np.mean(vels) + 1e-6)
        
        return metrics

    # =========================================================================
    # 2. PROCESAMIENTO CINEMÁTICO
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
        df['nx'], df['ny'], df['nz'] = vecs[:,0], vecs[:,1], vecs[:,2]
        
        dots = np.clip(np.sum(vecs[:-1] * vecs[1:], axis=1), -1.0, 1.0)
        ang_steps = np.insert(np.degrees(np.arccos(dots)), 0, 0)
        df['pos_deg'] = np.cumsum(ang_steps)
        
        w, p = self.cfg['PARAMS']['SAVGOL_WINDOW'], self.cfg['PARAMS']['SAVGOL_POLY']
        if len(df) <= w:
            df['velocidad'] = 0; df['aceleracion'] = 0; df['jerk'] = 0
        else:
            df['velocidad'] = savgol_filter(np.gradient(df['pos_deg'], df['time_s']), w, p)
            df['aceleracion'] = savgol_filter(np.gradient(df['velocidad'], df['time_s']), w, p)
            df['jerk'] = savgol_filter(np.gradient(df['aceleracion'], df['time_s']), w, p)
        
        return df

    # =========================================================================
    # 3. LÓGICA DE EXPERIMENTOS (MEJORADA)
    # =========================================================================

    def analizar_exp0(self, df):
        """[EXP 0] Estabilidad (Fijación)."""
        m = {}
        vecs = df[['nx','ny','nz']].values
        center = np.mean(vecs, axis=0); center /= np.linalg.norm(center)
        offsets = np.degrees(np.arccos(np.clip(np.dot(vecs, center), -1.0, 1.0)))
        
        m['Estabilidad_Std_Deg'] = np.std(offsets)
        m['Estabilidad_Media_Deg'] = np.mean(offsets)
        
        if df['time_s'].iloc[-1] - df['time_s'].iloc[0] > 2.0:
            v1 = np.mean(vecs[:10], axis=0)
            v2 = np.mean(vecs[-10:], axis=0)
            drift = np.degrees(np.arccos(np.clip(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)), -1.0, 1.0)))
            m['Drift_Total_Deg'] = drift
        else:
            m['Drift_Total_Deg'] = 0
            
        return m

    def analizar_exp1_2(self, df, df_stim, t_start, t_end):
        """[EXP 1 & 2] Sácadas - VERSIÓN MEJORADA."""
        m = {}
        acc = df['aceleracion'].values
        peaks, _ = find_peaks(acc, height=self.cfg['PARAMS']['UMBRAL_ACEL_SACADA'], distance=5)
        
        sacadas = []
        for p in peaks:
            end_search = np.where(acc[p:] < 0)[0]
            if len(end_search) > 0:
                end = p + end_search[0]
                seg = df.iloc[p:end+1]
                dur = seg['time_s'].iloc[-1] - seg['time_s'].iloc[0]
                
                if self.cfg['PARAMS']['MIN_DUR_SACADA_S'] <= dur <= self.cfg['PARAMS']['MAX_DUR_SACADA_S']:
                    amp = seg['pos_deg'].iloc[-1] - seg['pos_deg'].iloc[0]
                    t_peak = seg['velocidad'].idxmax()
                    asym = (df.loc[t_peak, 'time_s'] - seg['time_s'].iloc[0]) / dur if dur > 0 else 0.5
                    
                    sacadas.append({
                        'amp': abs(amp), 
                        'dur': dur, 
                        'peak_vel': seg['velocidad'].max(),
                        'asym': asym,
                        't_start': seg['time_s'].iloc[0]
                    })
        
        df_sac = pd.DataFrame(sacadas)
        m['Num_Saccades'] = len(df_sac)
        
        if not df_sac.empty:
            # Métricas básicas
            m['Mean_Saccade_Amp'] = df_sac['amp'].mean()
            m['Mean_Peak_Vel'] = df_sac['peak_vel'].mean()
            m['Mean_Saccade_Dur'] = df_sac['dur'].mean()
            m['Mean_Saccade_Asymmetry'] = df_sac['asym'].mean()
            m['Latency_Mean'] = self.calcular_latencia_reaccion(df_sac, df_stim, t_start, t_end)
            
            # 🔥 NUEVO: Main Sequence Completo
            m.update(self.analizar_main_sequence_completo(df_sac))
            
            # 🔥 NUEVO: Distribución de sacadas
            m.update(self.analizar_distribucion_sacadas(df_sac, t_start, t_end))
            
        else:
            # Rellenar con ceros
            m.update({
                'Mean_Saccade_Amp': 0, 'Mean_Peak_Vel': 0, 'Mean_Saccade_Dur': 0,
                'Mean_Saccade_Asymmetry': 0, 'Latency_Mean': 0
            })
            m.update(self.analizar_main_sequence_completo(pd.DataFrame()))
            m.update(self.analizar_distribucion_sacadas(pd.DataFrame(), t_start, t_end))

        return m

    def analizar_exp3(self, df, df_stim):
        """[EXP 3] Seguimiento Suave (Pursuit)."""
        m = {}
        if df_stim is None: return {'RMSE_Error':0, 'Ganancia_Velocidad':0}
        
        t_eye = df['time_s'].values
        t_stim = df_stim['relative_time_s'].values - self.cfg['PARAMS']['TIME_OFFSET_S']
        
        scale_x = self.cfg['PARAMS']['FOV_H'] / 1920 
        scale_y = (self.cfg['PARAMS']['FOV_H']*9/16) / 1080
        
        tx = np.interp(t_eye, t_stim, (df_stim['stimulus_x']-1920/2)*scale_x)
        ty = np.interp(t_eye, t_stim, (df_stim['stimulus_y']-1080/2)*scale_y)
        
        ex = (df['gaze_x']-df['gaze_x'].mean()) * self.cfg['PARAMS']['FOV_H']
        ey = (df['gaze_y']-df['gaze_y'].mean()) * (self.cfg['PARAMS']['FOV_H']*9/16)
        
        m['RMSE_Error'] = np.sqrt(np.mean((ex-tx)**2 + (ey-ty)**2))
        
        dt = np.diff(t_eye, prepend=t_eye[0]); dt[dt<=0] = 0.008
        vt = np.sqrt(np.diff(tx, prepend=tx[0])**2 + np.diff(ty, prepend=ty[0])**2) / dt
        
        with np.errstate(divide='ignore', invalid='ignore'):
            gain = df['velocidad'].values / vt
            gain = gain[(gain>0.1) & (gain<4.0)]
            
        m['Ganancia_Velocidad'] = np.mean(gain) if len(gain)>0 else 0
        return m

    # =========================================================================
    # 4. EJECUCIÓN CON VENTANAS DESLIZANTES
    # =========================================================================
    def run(self):
        print("="*70)
        print(f"🧬 PIPELINE BIOMÉTRICO MEJORADO - EXP {self.cfg['EXP_NUM']}")
        print(f"   ⏱ Ventana: {self.cfg['PARAMS']['WINDOW_SIZE_S']}s | Paso: {self.cfg['PARAMS']['WINDOW_STRIDE_S']}s")
        print(f"   🔥 NUEVO: Main Sequence Completo + Distribuciones")
        print("="*70)
        
        files = glob.glob(os.path.join(self.cfg['INPUT_DIR'], "*_data.csv"))
        dataset = []
        
        for f in files:
            if "Resumen" in f or "Features" in f or "METRICS" in f: continue
            vid_id = os.path.basename(f).replace("_data.csv", "")
            print(f"   • {vid_id}...", end=" ")
            
            try:
                df_raw = pd.read_csv(f)
                df_full = self.procesar_datos_raw(df_raw)
                
                if df_full is None: 
                    print("Skipped (Sin datos válidos)"); continue

                df_full = self.calcular_cinematica(df_full)
                df_stim = self.load_stimulus_log(vid_id)
                
                max_time = df_full['time_s'].max()
                current_time = 0
                windows_count = 0
                
                while current_time + self.cfg['PARAMS']['WINDOW_SIZE_S'] <= max_time:
                    t_start = current_time
                    t_end = current_time + self.cfg['PARAMS']['WINDOW_SIZE_S']
                    
                    df_win = df_full[(df_full['time_s'] >= t_start) & (df_full['time_s'] < t_end)].copy()
                    
                    expected_samples = self.cfg['PARAMS']['WINDOW_SIZE_S'] * self.cfg['PARAMS']['SAMPLING_RATE']
                    if len(df_win) < expected_samples * self.cfg['PARAMS']['MIN_DATA_IN_WINDOW']:
                        current_time += self.cfg['PARAMS']['WINDOW_STRIDE_S']
                        continue
                        
                    row = {
                        'VideoID': vid_id,
                        'Window_Idx': windows_count,
                        'Window_Start': t_start
                    }
                    
                    # Estadísticas Distribucionales
                    row['Vel_Mean'] = df_win['velocidad'].mean()
                    row['Vel_Std'] = df_win['velocidad'].std()
                    row['Vel_Skew'] = skew(df_win['velocidad'].values)
                    row['Vel_Kurtosis'] = kurtosis(df_win['velocidad'].values)
                    
                    # FFT y Ruido
                    row.update(self.calcular_fft_features(df_win['velocidad'].values))
                    row.update(self.calcular_perfil_ruido(df_win))
                    row['Vel_Entropy'] = self.calcular_entropia_aproximada(df_win['velocidad'].values[::4])
                    
                    # Métricas específicas del Experimento
                    exp = self.cfg['EXP_NUM']
                    exp_metrics = {}
                    
                    if exp == 0:
                        exp_metrics = self.analizar_exp0(df_win)
                    elif exp in [1, 2]:
                        exp_metrics = self.analizar_exp1_2(df_win, df_stim, t_start, t_end)
                    elif exp == 3:
                        exp_metrics = self.analizar_exp3(df_win, df_stim)
                    
                    row.update(exp_metrics)
                    dataset.append(row)
                    
                    current_time += self.cfg['PARAMS']['WINDOW_STRIDE_S']
                    windows_count += 1
                
                print(f"✅ {windows_count} ventanas extraídas")
                
            except Exception as e: 
                print(f"❌ Error: {e}")
                traceback.print_exc()

        if dataset:
            df_out = pd.DataFrame(dataset).fillna(0)
            tag = {0: "Fixation", 1: "Reading", 2: "Saccades", 3: "Pursuit"}.get(self.cfg['EXP_NUM'], "Full")
            out_path = os.path.join(self.cfg['OUTPUT_DIR'], f"{self.cfg['NAME']}_{tag}_WINDOWED_METRICS.csv")
            df_out.to_csv(out_path, index=False)
            print(f"\n🎉 Generado: {out_path}")
            print(f"📊 Total Muestras: {len(df_out)} | Columnas: {len(df_out.columns)}")
            
            # Mostrar métricas de Main Sequence extraídas
            mainseq_cols = [c for c in df_out.columns if 'MainSeq' in c or 'Amp_' in c or 'PeakVel_' in c or 'ISI_' in c]
            if mainseq_cols:
                print(f"\n🔥 MÉTRICAS DE MAIN SEQUENCE EXTRAÍDAS ({len(mainseq_cols)}):")
                for col in mainseq_cols:
                    mean_val = df_out[col].mean()
                    std_val = df_out[col].std()
                    print(f"   • {col:35s}: μ={mean_val:8.3f} | σ={std_val:8.3f}")

if __name__ == "__main__":
    sys = BiometricPipeline(CONFIG)
    sys.run()