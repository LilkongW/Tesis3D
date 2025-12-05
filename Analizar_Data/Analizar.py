import pandas as pd
import numpy as np
import os
import glob
import traceback
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress, skew, kurtosis
from scipy.fft import fft, fftfreq

# =============================================================================
#  CONFIGURACIÓN DEL SISTEMA
# =============================================================================
CONFIG = {
    'NAME': "Victor",
    'EXP_NUM': 1,  # <--- CAMBIAR AQUÍ: 0=Fijo, 1=Lectura, 2=Saltos, 3=Espiral
    'PATHS': {
        'BASE': r"C:\Users\Victor\Documents\Tesis3D" 
    },
    'PARAMS': {
        'SAMPLING_RATE': 120,     # Hz
        'FOV_H': 60,              # Grados
        'SAVGOL_WINDOW': 21,
        'SAVGOL_POLY': 3,
        'TIME_OFFSET_S': 0.600,   # <--- OFFSET DE 600ms
        
        # Umbrales Biométricos
        'UMBRAL_VEL_FIJACION': 100.0, 
        'UMBRAL_ACEL_SACADA': 250.0,
        'MIN_DUR_FIJACION_S': 0.250,    
        'MIN_DUR_SACADA_S': 0.020,      
        'MAX_DUR_SACADA_S': 0.200,
        'MIN_FRAMES_PARPADEO': 6,
        
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
        if N < 10: return {'Power_Low': 0, 'Power_High': 0}
        
        yf = fft(serie)
        xf = fftfreq(N, 1 / self.cfg['PARAMS']['SAMPLING_RATE'])
        power = np.abs(yf[:N//2])
        freqs = xf[:N//2]
        
        p_low = np.sum(power[(freqs > 0.1) & (freqs <= 4)])
        p_high = np.sum(power[(freqs > 4) & (freqs <= 20)])
        
        return {
            'Potencia_Freq_Baja': p_low,
            'Potencia_Freq_Alta': p_high,
            'Ratio_Freq_Alta_Baja': p_high / (p_low + 1e-6)
        }

    def calcular_latencia_reaccion(self, df_sacadas, df_stim):
        """Calcula tiempo de reacción ante estímulos."""
        if df_sacadas.empty or df_stim is None: return 0
        if 'relative_time_s' not in df_stim.columns: return 0
        
        df_stim['change'] = (df_stim['stimulus_x'] != df_stim['stimulus_x'].shift()) | \
                            (df_stim['stimulus_y'] != df_stim['stimulus_y'].shift())
        
        event_times = df_stim.loc[df_stim['change'], 'relative_time_s'].values
        # Ajustar offset
        event_times = event_times - self.cfg['PARAMS']['TIME_OFFSET_S']
        event_times = event_times[event_times >= 0]
        
        if len(event_times) == 0: return 0
        
        latencias = []
        sac_starts = df_sacadas['t_start'].values
        
        for t_stim in event_times:
            min_t = t_stim + self.cfg['PARAMS']['LATENCY_MIN_S']
            max_t = t_stim + self.cfg['PARAMS']['LATENCY_MAX_S']
            
            matches = sac_starts[(sac_starts >= min_t) & (sac_starts <= max_t)]
            if len(matches) > 0:
                latencias.append(matches[0] - t_stim)
                
        return np.mean(latencias) if latencias else 0

    # =========================================================================
    # 2. PROCESAMIENTO CINEMÁTICO (OFFSET + VECTORES 3D)
    # =========================================================================

    def procesar_datos_raw(self, df_raw):
        """Limpia datos y aplica el Offset de tiempo (Sin parpadeos)."""
        df = df_raw.copy()
        
        if 'valid_deteccion' in df.columns:
            df = df[df['valid_deteccion'].astype(str) == 'True'].copy()
            
        if df.empty: return None

        t0 = df['timestamp_ms'].iloc[0]
        df['time_s'] = ((df['timestamp_ms'] - t0) / 1000.0) - self.cfg['PARAMS']['TIME_OFFSET_S']
        
        # Eliminar datos pre-offset
        df = df[df['time_s'] >= 0].copy()
        
        if len(df) < 50: return None
        return df

    def calcular_cinematica(self, df):
        """Cálculo vectorial 3D + Derivadas."""
        vecs = df[['gaze_x', 'gaze_y', 'gaze_z']].values
        normas = np.linalg.norm(vecs, axis=1, keepdims=True); normas[normas==0]=1.0
        vecs = vecs / normas
        df['nx'], df['ny'], df['nz'] = vecs[:,0], vecs[:,1], vecs[:,2]
        
        dots = np.clip(np.sum(vecs[:-1] * vecs[1:], axis=1), -1.0, 1.0)
        ang_steps = np.insert(np.degrees(np.arccos(dots)), 0, 0)
        df['pos_deg'] = np.cumsum(ang_steps)
        
        w, p = self.cfg['PARAMS']['SAVGOL_WINDOW'], self.cfg['PARAMS']['SAVGOL_POLY']
        df['velocidad'] = savgol_filter(np.gradient(df['pos_deg'], df['time_s']), w, p)
        df['aceleracion'] = savgol_filter(np.gradient(df['velocidad'], df['time_s']), w, p)
        df['jerk'] = savgol_filter(np.gradient(df['aceleracion'], df['time_s']), w, p)
        
        return df

    # =========================================================================
    # 3. LÓGICA DE EXPERIMENTOS
    # =========================================================================

    def analizar_exp0(self, df):
        """[EXP 0] Estabilidad Vectorial."""
        m = {}
        vecs = df[['nx','ny','nz']].values
        center = np.mean(vecs, axis=0); center /= np.linalg.norm(center)
        offsets = np.degrees(np.arccos(np.clip(np.dot(vecs, center), -1.0, 1.0)))
        
        m['Estabilidad_Std_Deg'] = np.std(offsets)
        m['Estabilidad_Media_Deg'] = np.mean(offsets)
        
        v1 = np.mean(vecs[df['time_s']<1.0], axis=0)
        v2 = np.mean(vecs[df['time_s']>df['time_s'].iloc[-1]-1.0], axis=0)
        if np.linalg.norm(v1)>0 and np.linalg.norm(v2)>0:
            drift = np.degrees(np.arccos(np.clip(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)), -1.0, 1.0)))
            m['Drift_Total_Deg'] = drift
        else: m['Drift_Total_Deg'] = 0
        
        # Rellenar con 0 métricas de otros exps
        keys_sac = ['Num_Saccades','Mean_Saccade_Amp','Mean_Peak_Vel','Mean_Peak_Acc',
                    'MainSeq_Slope','MainSeq_Intercept','MainSeq_R2',
                    'LinearSeq_Slope','LinearSeq_Intercept','LinearSeq_R2', # <--- AQUÍ ESTÁN
                    'Mean_Saccade_Jerk','Latency_Mean',
                    'Num_Fixations','Mean_Fixation_Dur','Mean_Fixation_Dispersion',
                    'RMSE_Error','Ganancia_Velocidad']
        for k in keys_sac: m[k] = 0

        m.update(self.calcular_fft_features(df['velocidad'].values))
        m['Vel_Entropy'] = self.calcular_entropia_aproximada(df['velocidad'].values[::5])
        return m

    def analizar_exp1_2(self, df, df_stim):
        """[EXP 1 & 2] Sácadas Completas (Incluye K e Intercepto)."""
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
                        'peak_acc': seg['aceleracion'].abs().max(),
                        'mean_jerk': seg['jerk'].abs().mean(), 
                        'asym': asym,
                        't_start': seg['time_s'].iloc[0]
                    })
        
        # Fijaciones
        df['es_fij'] = df['velocidad'] < self.cfg['PARAMS']['UMBRAL_VEL_FIJACION']
        grps = (df['es_fij'] != df['es_fij'].shift()).cumsum()
        fijaciones = []
        for _, g in df[df['es_fij']].groupby(grps):
            dur = g['time_s'].iloc[-1] - g['time_s'].iloc[0]
            if dur >= self.cfg['PARAMS']['MIN_DUR_FIJACION_S']:
                vecs = g[['nx','ny','nz']].values
                disp = np.std(np.degrees(np.arccos(np.clip(np.dot(vecs, np.mean(vecs,0)/np.linalg.norm(np.mean(vecs,0))), -1, 1))))
                fijaciones.append({'dur': dur, 'disp': disp})

        # Métricas
        df_sac = pd.DataFrame(sacadas)
        if not df_sac.empty:
            m['Num_Saccades'] = len(df_sac)
            m['Mean_Saccade_Amp'] = df_sac['amp'].mean()
            m['Std_Saccade_Amp'] = df_sac['amp'].std()
            m['Mean_Saccade_Dur'] = df_sac['dur'].mean()
            m['Mean_Peak_Vel'] = df_sac['peak_vel'].mean()
            m['Mean_Peak_Acc'] = df_sac['peak_acc'].mean()
            m['Mean_Saccade_Jerk'] = df_sac['mean_jerk'].mean()
            m['Mean_Saccade_Asymmetry'] = df_sac['asym'].mean()
            
            m['Latency_Mean'] = self.calcular_latencia_reaccion(df_sac, df_stim)
            
            if len(df_sac) > 3:
                # Main Sequence: Vel_Pico = K * Amp + C
                s, i, r, _, _ = linregress(df_sac['amp'], df_sac['peak_vel'])
                m['MainSeq_Slope'] = s
                m['MainSeq_Intercept'] = i
                m['MainSeq_R2'] = r**2
                
                # Linear Sequence: Duracion = M * Amp + C
                s2, i2, r2, _, _ = linregress(df_sac['amp'], df_sac['dur'])
                m['LinearSeq_Slope'] = s2
                m['LinearSeq_Intercept'] = i2  # <--- AGREGADO AQUÍ
                m['LinearSeq_R2'] = r2**2
            else:
                m.update({'MainSeq_Slope':0, 'MainSeq_Intercept':0, 'MainSeq_R2':0, 
                          'LinearSeq_Slope':0, 'LinearSeq_Intercept':0, 'LinearSeq_R2':0})
        else:
            for k in ['Num_Saccades','Mean_Saccade_Amp','Mean_Peak_Vel','Mean_Peak_Acc',
                      'MainSeq_Slope','MainSeq_Intercept','MainSeq_R2',
                      'LinearSeq_Slope','LinearSeq_Intercept','LinearSeq_R2',
                      'Mean_Saccade_Jerk','Latency_Mean']: m[k]=0

        if fijaciones:
            df_fij = pd.DataFrame(fijaciones)
            m['Num_Fixations'] = len(fijaciones)
            m['Mean_Fixation_Dur'] = df_fij['dur'].mean()
            m['Std_Fixation_Dur'] = df_fij['dur'].std()
            m['Mean_Fixation_Dispersion'] = df_fij['disp'].mean()
        else: m.update({'Num_Fixations':0, 'Mean_Fixation_Dur':0, 'Mean_Fixation_Dispersion':0})
        
        # Rellenar métricas de otros exps
        for k in ['Estabilidad_Std_Deg', 'Drift_Total_Deg', 'RMSE_Error', 'Ganancia_Velocidad']: m[k] = 0

        m.update(self.calcular_fft_features(df['velocidad'].values))
        m['Vel_Entropy'] = self.calcular_entropia_aproximada(df['velocidad'].values[::5])
        
        return m

    def analizar_exp3(self, df, df_stim):
        """[EXP 3] Espiral (RMSE + Ganancia)."""
        m = {}
        if df_stim is None: return {'RMSE':0, 'Ganancia':0}
        
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
        with np.errstate(divide='ignore'):
            gain = df['velocidad'].values / vt
            gain = gain[(gain>0.1) & (gain<4.0)]
        m['Ganancia_Velocidad'] = np.mean(gain) if len(gain)>0 else 0
        
        # Rellenar con 0 métricas de otros exps
        keys_sac = ['Num_Saccades','Mean_Saccade_Amp','Mean_Peak_Vel','Mean_Peak_Acc',
                    'MainSeq_Slope','MainSeq_Intercept','MainSeq_R2',
                    'LinearSeq_Slope','LinearSeq_Intercept','LinearSeq_R2',
                    'Mean_Saccade_Jerk','Latency_Mean',
                    'Num_Fixations','Mean_Fixation_Dur','Mean_Fixation_Dispersion',
                    'Estabilidad_Std_Deg', 'Drift_Total_Deg']
        for k in keys_sac: m[k] = 0

        m.update(self.calcular_fft_features(df['velocidad'].values))
        m['Vel_Entropy'] = self.calcular_entropia_aproximada(df['velocidad'].values[::5])
        return m

    # =========================================================================
    # 4. EJECUCIÓN
    # =========================================================================
    def run(self):
        print("="*70)
        print(f"🧬 PIPELINE BIOMÉTRICO (FULL METRICS + OFFSET 600ms) - EXP {self.cfg['EXP_NUM']}")
        print("="*70)
        
        files = glob.glob(os.path.join(self.cfg['INPUT_DIR'], "*_data.csv"))
        dataset = []
        
        for f in files:
            if "Resumen" in f or "Features" in f or "METRICS" in f: continue
            vid_id = os.path.basename(f).replace("_data.csv", "")
            print(f"   • {vid_id}...", end=" ")
            
            try:
                # 1. Cargar y Offset
                df_raw = pd.read_csv(f)
                df = self.procesar_datos_raw(df_raw)
                
                if df is None: print("Skipped (Pocos datos tras offset)"); continue
                
                # 2. Cinemática
                df = self.calcular_cinematica(df)
                
                # 3. Distribución
                dist_metrics = {
                    'Vel_Global_Mean': df['velocidad'].mean(),
                    'Vel_Global_Std': df['velocidad'].std(),
                    'Vel_Skewness': skew(df['velocidad'].values),
                    'Vel_Kurtosis': kurtosis(df['velocidad'].values)
                }
                
                # 4. Experimentos
                exp_metrics = {}
                exp = self.cfg['EXP_NUM']
                df_stim = self.load_stimulus_log(vid_id)
                
                if exp == 0:
                    exp_metrics = self.analizar_exp0(df)
                    print(f"✅ Disp: {exp_metrics.get('Estabilidad_Std_Deg',0):.2f}°")
                elif exp in [1, 2]:
                    exp_metrics = self.analizar_exp1_2(df, df_stim)
                    print(f"✅ Sac: {exp_metrics.get('Num_Saccades',0)} | Acc: {exp_metrics.get('Mean_Peak_Acc',0):.0f}")
                elif exp == 3:
                    exp_metrics = self.analizar_exp3(df, df_stim)
                    print(f"✅ RMSE: {exp_metrics.get('RMSE_Error',0):.2f}")
                
                row = {'VideoID': vid_id}
                row.update(dist_metrics)
                row.update(exp_metrics)
                dataset.append(row)
                
            except Exception as e: 
                print(f"❌ {e}")
                traceback.print_exc()

        if dataset:
            df_out = pd.DataFrame(dataset).fillna(0)
            tag = {0: "Fixation", 1: "Reading", 2: "Saccades", 3: "Pursuit"}.get(self.cfg['EXP_NUM'], "Full")
            out_path = os.path.join(self.cfg['OUTPUT_DIR'], f"{self.cfg['NAME']}_{tag}_METRICS.csv")
            df_out.to_csv(out_path, index=False)
            print(f"\n🎉 Generado: {out_path}")
            print(f"📊 Columnas: {len(df_out.columns)}")

if __name__ == "__main__":
    sys = BiometricPipeline(CONFIG)
    sys.run()