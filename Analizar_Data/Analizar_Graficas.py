import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
from scipy.signal import savgol_filter, find_peaks

# Ignorar warnings
warnings.filterwarnings('ignore')

# Obtener la ruta base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# =============================================================================
#  CONFIGURACIÓN (Heredada de tu script original)
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
        
        # Umbrales (Los mismos que usas para las métricas)
        'UMBRAL_VEL_FIJACION': 100.0,
        'UMBRAL_ACEL_SACADA': 200.0, # Umbral clave para tu detección
        'MIN_DUR_FIJACION_S': 0.100,
        'MIN_DUR_SACADA_S': 0.008,      
        'MAX_DUR_SACADA_S': 0.200,
    }
}

CONFIG['EXP_DIR'] = os.path.join(CONFIG['PATHS']['BASE'], "Data", f"Experimento_{CONFIG['EXP_NUM']}")
OUTPUT_GRAPH_DIR = os.path.join(CONFIG['PATHS']['BASE'], "Analisis_Avanzados", "Visualizacion_Eventos")
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

class EventVisualizerPipeline:
    def __init__(self, config):
        self.cfg = config
        
    # =========================================================================
    # 1. PROCESAMIENTO (Copiado EXACTO de tu Generar_Metricas.py)
    # =========================================================================
    
    def procesar_datos_raw(self, df_raw):
        """Limpieza inicial y ajuste de tiempo"""
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
        """Cálculo de velocidad y aceleración angular (Tu algoritmo original)"""
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

    # =========================================================================
    # 2. DETECCIÓN DE EVENTOS (Adaptado para devolver tiempos para graficar)
    # =========================================================================
    
    def obtener_intervalos_eventos(self, df):
        """
        Detecta los tiempos de inicio y fin de Sacadas y Fijaciones
        usando tu lógica de picos de aceleración.
        """
        acc = df['aceleracion'].values
        # Tu lógica original: picos en aceleración
        peaks, _ = find_peaks(acc, height=self.cfg['PARAMS']['UMBRAL_ACEL_SACADA'], distance=5)
        
        sacadas_intervals = [] # Lista de tuplas (t_start, t_end)
        
        # 1. Detectar Sacadas
        for p in peaks:
            # Buscar cruce por cero después del pico (fin de la sacada)
            end_search = np.where(acc[p:] < 0)[0]
            if len(end_search) > 0:
                end = p + end_search[0]
                seg = df.iloc[p:end+1]
                dur = seg['time_s'].iloc[-1] - seg['time_s'].iloc[0]
                
                # Filtros de duración
                if self.cfg['PARAMS']['MIN_DUR_SACADA_S'] <= dur <= self.cfg['PARAMS']['MAX_DUR_SACADA_S']:
                    t_start = seg['time_s'].iloc[0]
                    t_end = seg['time_s'].iloc[-1]
                    sacadas_intervals.append((t_start, t_end))
        
        # 2. Detectar Fijaciones (Espacios entre sacadas)
        fijaciones_intervals = []
        if len(sacadas_intervals) > 1:
            # Ordenar por tiempo
            sacadas_intervals.sort(key=lambda x: x[0])
            
            for i in range(len(sacadas_intervals)-1):
                sac_end = sacadas_intervals[i][1]
                sac_next_start = sacadas_intervals[i+1][0]
                dur_fix = sac_next_start - sac_end
                
                if dur_fix > self.cfg['PARAMS']['MIN_DUR_FIJACION_S']:
                    fijaciones_intervals.append((sac_end, sac_next_start))
                    
        return sacadas_intervals, fijaciones_intervals

    # =========================================================================
    # 3. VISUALIZACIÓN
    # =========================================================================
    
    def graficar_eventos(self, df, sacadas, fijaciones, participant, video_id):
        """
        Genera gráfico dual: Velocidad y Aceleración con eventos sombreados.
        """
        # Limitar tamaño si es muy largo para que se vea bien (opcional, primeros 10s)
        # O graficar todo. Aquí graficamos todo pero con un ancho grande.
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        
        times = df['time_s']
        vel = df['velocidad']
        acc = df['aceleracion']
        
        # --- GRAFICO 1: VELOCIDAD ---
        ax1.plot(times, vel, color='#333333', linewidth=1, label='Velocidad (deg/s)')
        ax1.set_ylabel('Velocidad (°/s)')
        ax1.set_title(f'Detección de Eventos: {participant} - {video_id}', fontsize=14, fontweight='bold')
        
        # --- GRAFICO 2: ACELERACIÓN ---
        ax2.plot(times, acc, color='#1f77b4', linewidth=1, label='Aceleración (deg/s²)')
        ax2.set_ylabel('Aceleración (°/s²)')
        ax2.set_xlabel('Tiempo (s)')
        
        # --- PINTAR EVENTOS ---
        # Sacadas (Rojo)
        label_added = False
        for (start, end) in sacadas:
            lbl = 'Sacada Detectada' if not label_added else None
            # Pintar en ambos gráficos
            ax1.axvspan(start, end, color='red', alpha=0.3, label=lbl)
            ax2.axvspan(start, end, color='red', alpha=0.3)
            label_added = True
            
        # Fijaciones (Verde)
        label_added = False
        for (start, end) in fijaciones:
            lbl = 'Fijación' if not label_added else None
            ax1.axvspan(start, end, color='green', alpha=0.15, label=lbl)
            ax2.axvspan(start, end, color='green', alpha=0.15)
            label_added = True

        # Líneas de referencia
        ax2.axhline(self.cfg['PARAMS']['UMBRAL_ACEL_SACADA'], color='orange', linestyle='--', alpha=0.5, label='Umbral Sacada')
        
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Guardar
        filename = f"Visual_Eventos_{participant}_{video_id}.png"
        save_path = os.path.join(OUTPUT_GRAPH_DIR, filename)
        plt.savefig(save_path, dpi=150)
        plt.close()

    # =========================================================================
    # 4. RUN LOOP
    # =========================================================================

    def run(self):
        print("="*70)
        print(f"👁️ VISUALIZADOR DE EVENTOS (VELOCIDAD Y ACELERACIÓN)")
        print(f"📂 Origen de datos: {self.cfg['EXP_DIR']}")
        print(f"📂 Salida de gráficas: {OUTPUT_GRAPH_DIR}")
        print("="*70)
        
        participant_dirs = [d for d in os.listdir(self.cfg['EXP_DIR']) 
                           if os.path.isdir(os.path.join(self.cfg['EXP_DIR'], d)) and d.endswith('_data')]
        
        count = 0
        for participant_dir in participant_dirs:
            participant_name = participant_dir.replace('_data', '')
            participant_path = os.path.join(self.cfg['EXP_DIR'], participant_dir)
            
            print(f"\n🔍 Analizando Participante: {participant_name}")
            files = glob.glob(os.path.join(participant_path, "*_data.csv"))
            
            for f in files:
                if "Resumen" in f or "Features" in f: continue
                vid_id = os.path.basename(f).replace("_data.csv", "")
                
                try:
                    # 1. Cargar
                    df_raw = pd.read_csv(f)
                    df_full = self.procesar_datos_raw(df_raw)
                    
                    if df_full is None: 
                        print(f"   ⚠️ {vid_id}: Datos insuficientes o inválidos.")
                        continue

                    # 2. Calcular Cinemática (Velocidad/Aceleración)
                    df_full = self.calcular_cinematica(df_full)
                    
                    # 3. Detectar Eventos (Obtener intervalos de tiempo)
                    sacadas, fijaciones = self.obtener_intervalos_eventos(df_full)
                    
                    # 4. Generar Gráfico
                    if len(sacadas) > 0:
                        self.graficar_eventos(df_full, sacadas, fijaciones, participant_name, vid_id)
                        print(f"   📸 Gráfico generado: {vid_id} ({len(sacadas)} sacadas)")
                        count += 1
                    else:
                        print(f"   ⚠️ {vid_id}: No se detectaron sacadas claras.")
                    
                except Exception as e:
                    print(f"   ❌ Error procesando {vid_id}: {str(e)}")
        
        print("\n" + "="*70)
        print(f"✅ FINALIZADO. {count} gráficos generados.")

if __name__ == "__main__":
    pipeline = EventVisualizerPipeline(CONFIG)
    pipeline.run()