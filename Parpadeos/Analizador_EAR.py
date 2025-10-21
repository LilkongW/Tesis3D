import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import glob 
from scipy.signal import savgol_filter 
import pandas as pd 

# ==============================================================================
# CLASE EARAnalyzer 
# ==============================================================================

class EARAnalyzer:
    """
    Clase adaptada para calcular la Relación de Aspecto del Ojo (EAR) mejorada
    utilizando puntos de referencia de MediaPipe.
    """
    
    RIGHT_EYE_VERTICALS = [(160, 144), (159, 145), (158, 153), (157, 154), (173, 155)]
    RIGHT_EYE_HORIZONTAL = (33, 133)
    
    LEFT_EYE_VERTICALS = [(387, 373), (386, 374), (385, 380), (384, 381), (398, 382)]
    LEFT_EYE_HORIZONTAL = (362, 263)

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def improved_ear(self, vertical_pairs, horizontal_pair, landmarks_dict):
        """Calcular EAR mejorado."""
        try:
            verticals = []
            for top, bottom in vertical_pairs:
                if top in landmarks_dict and bottom in landmarks_dict:
                    vertical_dist = np.linalg.norm(
                        np.array(landmarks_dict[top]) - np.array(landmarks_dict[bottom])
                    )
                    verticals.append(vertical_dist)
            
            if not verticals: return 0
            avg_vertical = np.mean(verticals)
            
            if horizontal_pair[0] not in landmarks_dict or horizontal_pair[1] not in landmarks_dict: return 0
                
            horizontal_dist = np.linalg.norm(
                np.array(landmarks_dict[horizontal_pair[0]]) - np.array(landmarks_dict[horizontal_pair[1]])
            )
            
            if horizontal_dist == 0: return 0
            
            return avg_vertical / horizontal_dist
            
        except (KeyError, TypeError, ZeroDivisionError):
            return 0

    def calculate_ear_for_frame(self, frame):
        """Procesar un frame y calcular el EAR promedio."""
        if frame is None: return None
            
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            landmarks_dict = {i: (landmark.x * w, landmark.y * h) for i, landmark in enumerate(face_landmarks.landmark)}
            
            right_ear = self.improved_ear(self.RIGHT_EYE_VERTICALS, self.RIGHT_EYE_HORIZONTAL, landmarks_dict)
            left_ear = self.improved_ear(self.LEFT_EYE_VERTICALS, self.LEFT_EYE_HORIZONTAL, landmarks_dict)
            
            if right_ear > 0 and left_ear > 0:
                return (right_ear + left_ear) / 2.0
        
        return None

# ------------------------------------------------------------------------------
# FUNCIÓN DE PROCESAMIENTO DE UNA SOLA PASADA (Ajustada para MIN_DURATION)
# ------------------------------------------------------------------------------

def analyze_video_data(video_path, ear_analyzer, data_storage, phase=1):
    """
    Procesa un solo video, calcula EAR crudo, lo suaviza y detecta parpadeos 
    usando el umbral de aceleración POSITIVA y una duración mínima en fotogramas.
    """
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"⚠ Error: No se pudo abrir el video {os.path.basename(video_path)}")
        return float('inf'), float('-inf')

    ear_values = []
    timestamps = []
    frame_count = 0
    
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 30 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 

        ear = ear_analyzer.calculate_ear_for_frame(frame)
        
        if ear is not None:
            ear_values.append(ear)
            time_in_seconds = frame_count / fps
            timestamps.append(time_in_seconds)
        
        frame_count += 1
        
        if phase == 1 and frame_count % 300 == 0:
              print(f"   -> Recopilando: {frame_count} frames...")
            
    cap.release()
    
    if not ear_values:
        return float('inf'), float('-inf')
    
    ear_values_array = np.array(ear_values)
    
    # ----------------------------------------------------
    # 1. APLICAR FILTRO SAVITZKY-GOLAY (SUAVIZADO)
    # ----------------------------------------------------
    
    WINDOW_LENGTH = 7 
    POLY_ORDER = 3
    
    if len(ear_values_array) >= WINDOW_LENGTH:
        smoothed_ear = savgol_filter(ear_values_array, window_length=WINDOW_LENGTH, polyorder=POLY_ORDER)
    else:
        smoothed_ear = ear_values_array
        
    # ----------------------------------------------------
    # 2. CALCULAR DERIVADAS (Velocidad y Aceleración)
    # ----------------------------------------------------
    
    # Derivada 1 (Velocidad): d(EAR)/dt
    ear_diff = np.diff(smoothed_ear, prepend=smoothed_ear[0])
    
    # Derivada 2 (Aceleración): d^2(EAR)/dt^2
    ear_accel = np.diff(ear_diff, prepend=ear_diff[0])

    # ----------------------------------------------------
    # 3. DETECCIÓN DE PARPADEO (Criterio: Aceleración + Duración Mínima)
    # ----------------------------------------------------
    
    BLINK_LEVEL_THRESHOLD = 0.20 
    ACCELERATION_THRESHOLD = 0.008

    # *** CAMBIO PRINCIPAL: Establecer la duración mínima a 2 fotogramas ***
    MIN_BLINK_DURATION_FRAMES = 3 
    # *******************************************************************

    # El parpadeo es donde la aceleración POSITIVA está por encima del umbral
    is_blink_candidate = ear_accel > ACCELERATION_THRESHOLD
    
    blink_events = []
    
    # CORRECCIÓN: Inicializar array de marcadores de parpadeo
    blink_frame_marker = np.zeros(len(smoothed_ear), dtype=int)
    
    i = 0
    
    # *** LÓGICA DE CONSOLIDACIÓN CON CRITERIO DE DURACIÓN MÍNIMA (CORREGIDA) ***
    while i < len(smoothed_ear):
        if is_blink_candidate[i]:
            start_index = i
            
            # 1. Encontrar el final de la ráfaga (consecutiva por encima del umbral)
            while i < len(smoothed_ear) and is_blink_candidate[i]:
                i += 1
            end_index = i - 1
            
            # 2. Calcular la duración de la ráfaga
            duration = end_index - start_index + 1
            
            # 3. Aplicar el criterio de duración mínima
            if duration >= MIN_BLINK_DURATION_FRAMES:
                # CORRECCIÓN CLAVE: Marcar todos los frames del evento, no solo el inicio
                for frame_idx in range(start_index, end_index + 1):
                    blink_frame_marker[frame_idx] = 1
                    
                blink_events.append({
                    'start_time': timestamps[start_index],
                    'end_time': timestamps[end_index]
                })
        else:
            i += 1
    # *** FIN LÓGICA DE CONSOLIDACIÓN ***
    
    # Duración total del video en segundos
    video_duration_sec = timestamps[-1] if timestamps else 0
    total_blinks = len(blink_events)
    # Frecuencia de parpadeo (parpadeos por minuto)
    blink_frequency_bpm = (total_blinks / video_duration_sec) * 60 if video_duration_sec > 0 else 0

    # ----------------------------------------------------
    # 4. ALMACENAR DATOS
    # ----------------------------------------------------
    
    data_storage[video_path] = {
        'raw_ear': ear_values_array,
        'smoothed_ear': smoothed_ear,
        'ear_diff': ear_diff,
        'ear_accel': ear_accel,
        'time': timestamps,
        'blinks': blink_events,
        'blink_frame_marker': blink_frame_marker, # Para marcar los frames de parpadeo en el CSV
        'video_fps': fps,
        'total_frames': len(ear_values_array),
        'total_blinks': total_blinks,
        'blink_frequency_bpm': blink_frequency_bpm,
        'blink_thresholds': {
            'level': BLINK_LEVEL_THRESHOLD, 
            'acceleration': ACCELERATION_THRESHOLD
        },
        'min_duration_frames': MIN_BLINK_DURATION_FRAMES # Almacenar duración mínima
    }
    
    return np.min(ear_values_array), np.max(ear_values_array)

# ------------------------------------------------------------------------------
# NUEVA FUNCIÓN: GUARDAR DATOS A CSV (CORREGIDA)
# ------------------------------------------------------------------------------

def save_data_to_csv(data, video_filename, output_dir):
    """
    Guarda los datos de EAR por frame, las derivadas, el tiempo y las métricas
    en un archivo CSV.
    """
    try:
        # Crear DataFrame con datos por frame
        # CORRECCIÓN: Convertir explícitamente a booleano para mejor legibilidad
        blink_detected = data['blink_frame_marker'].astype(bool)
        
        df = pd.DataFrame({
            'Tiempo_s': data['time'],
            'Frame_Index': range(len(data['time'])),
            'EAR_Crudo': data['raw_ear'],
            'EAR_Suavizado': data['smoothed_ear'],
            'Velocidad_dEAR_dt': data['ear_diff'],
            'Aceleracion_d2EAR_dt2': data['ear_accel'],
            'Blink_Detectado': blink_detected  # Ahora como booleano True/False
        })
        
        # Generar nombre de archivo
        safe_name = video_filename.replace('.', '_')
        csv_path = os.path.join(output_dir, f"Raw_{safe_name}.csv")
        
        # Guardar el DataFrame
        df.to_csv(csv_path, index=False, float_format='%.6f')
        
        # Mostrar estadísticas de detección para verificar
        total_blink_frames = blink_detected.sum()
        total_frames = len(blink_detected)
        print(f"   📊 Frames con parpadeo detectado: {total_blink_frames}/{total_frames}")
        
        # --- Guardar un archivo de métricas resumidas ---
        metrics_data = {
            'Video': [video_filename],
            'Duracion_Total_s': [data['time'][-1] if data['time'] else 0],
            'Total_Frames': [data['total_frames']],
            'Frames_Con_Parpadeo': [total_blink_frames],
            'FPS': [data['video_fps']],
            'EAR_Minimo_Raw': [data['raw_ear'].min()],
            'EAR_Maximo_Raw': [data['raw_ear'].max()],
            'Total_Parpadeos_Detectados': [data['total_blinks']],
            'Frecuencia_Parpadeo_BPM': [data['blink_frequency_bpm']],
            'Umbral_Aceleracion': [data['blink_thresholds']['acceleration']],
            'Min_Duracion_Frames': [data['min_duration_frames']]
        }
        df_metrics = pd.DataFrame(metrics_data)
        metrics_csv_path = os.path.join(output_dir, f"Resumen_Metricas_{safe_name}.csv")
        df_metrics.to_csv(metrics_csv_path, index=False, float_format='%.4f')

        print(f"   💾 Datos por frame (CSV) guardados en: {csv_path}")
        print(f"   📊 Métricas resumidas (CSV) guardadas en: {metrics_csv_path}")

    except Exception as e:
        print(f"   ⚠ Error al guardar CSV para {video_filename}: {str(e)}")


# ------------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ------------------------------------------------------------------------------

def process_videos_in_folder(videos_directory, output_dir):
    """
    Implementa la lógica de 2 fases: 1) Encontrar EAR global, 2) Graficar y guardar CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    ear_analyzer = EARAnalyzer()
    
    video_patterns = ['*.mp4', '*.avi', '*.mov', '*.webm']
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(videos_directory, pattern)))
    
    if not video_files:
        print(f"⚠️ No se encontraron archivos de video compatibles en la carpeta: {videos_directory}")
        return

    print(f"🎬 Encontrados {len(video_files)} videos para procesar.")
    
    # --- FASE 1: Recopilación y Búsqueda de Extremos Globales ---
    print("\n--- FASE 1: Recopilación, Suavizado y Búsqueda de Extremos Globales ---")
    
    all_video_data = {} 
    ear_global_min = float('inf')
    ear_global_max = float('-inf')

    for i, video_path in enumerate(video_files):
        video_filename = os.path.basename(video_path)
        print(f"[{i+1}/{len(video_files)}] Analizando {video_filename}...")
        
        min_local, max_local = analyze_video_data(video_path, ear_analyzer, all_video_data, phase=1)
        
        ear_global_min = min(ear_global_min, min_local)
        ear_global_max = max(ear_global_max, max_local)
        
    
    if not all_video_data:
        print("No se encontraron datos EAR válidos en ningún video. Terminando.")
        return
        
    print("\n--- Resultados de la Fase 1 ---")
    print(f"EAR Global Mínimo (Mínimo absoluto de cierre - RAW): {ear_global_min:.4f}")
    print(f"EAR Global Máximo (Máximo absoluto de apertura - RAW): {ear_global_max:.4f}")
    print("-" * 50)
    
    # --- FASE 2: Graficación y CSV ---
    print("\n--- FASE 2: Generación de Gráficos de Comparación (0-0.6) y CSV ---")
    
    for i, video_path in enumerate(video_files):
        video_filename = os.path.basename(video_path)
        
        if video_path in all_video_data:
            data = all_video_data[video_path]
            print(f"[{i+1}/{len(video_files)}] Procesando archivos: {video_filename}...")
            
            # Llamada a la función de guardado de CSV
            save_data_to_csv(data, video_filename, output_dir)
            
            # Llamada a la función de graficado (sin cambios)
            generate_comparison_plots(
                data['time'], 
                data['raw_ear'], 
                data['smoothed_ear'], 
                data['ear_diff'],
                data['ear_accel'], 
                data['blinks'],
                video_filename, 
                output_dir,
                data['blink_thresholds'],
                data['min_duration_frames'] 
            )
        else:
            print(f"[{i+1}/{len(video_files)}] Saltando {video_filename} (No se encontraron datos EAR).")


    print("🎉 ¡Procesamiento por lotes finalizado!")

# ------------------------------------------------------------------------------
# FUNCIÓN DE VISUALIZACIÓN MODIFICADA (sin cambios)
# ------------------------------------------------------------------------------

def generate_comparison_plots(timestamps, raw_ear, smoothed_ear, ear_diff, ear_accel, blinks, video_filename, output_dir, thresholds, min_duration_frames):
    """
    Genera una figura con cuatro subplots: EAR, Derivada (Velocidad), Segunda Derivada (Aceleración) 
    y Marcado de Parpadeos.
    """
    if not len(raw_ear): return 
    
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

        # Cuatro subplots: 4 filas, 1 columna
        fig, (ax1, ax2, ax_accel, ax3) = plt.subplots(4, 1, figsize=(15, 15), sharex=True) 
        
        # Ajustar el título para reflejar el criterio de duración mínima
        fig.suptitle(f'Análisis de EAR, Velocidad y Aceleración | Video: {video_filename}', fontsize=16)
        
        # --- Umbrales ---
        level_thresh = thresholds['level']
        accel_thresh = thresholds['acceleration']

        # ------------------------------------------------
        # SUBPLOT 1: EAR CRUDO y SUAVIZADO
        # ------------------------------------------------
        ax1.plot(timestamps, raw_ear, 'b-', linewidth=0.8, alpha=0.5, label='EAR Crudo')
        ax1.plot(timestamps, smoothed_ear, 'k-', linewidth=1.5, alpha=0.9, label='EAR Suavizado (SG)')
        ax1.set_title('1. Relación de Aspecto del Ojo (EAR)', fontsize=12)
        ax1.set_ylabel('Valor EAR', fontsize=10)
        ax1.grid(True, alpha=0.5, linestyle='--')
        ax1.set_ylim(0, 0.6)
        ax1.axhline(y=level_thresh, color='orange', linestyle='--', linewidth=1, label=f"EAR de Referencia (= {level_thresh:.2f})")
        ax1.legend(loc='upper right')

        # ------------------------------------------------
        # SUBPLOT 2: DERIVADA (VELOCIDAD)
        # ------------------------------------------------
        ax2.plot(timestamps, ear_diff, 'm-', linewidth=1.5, alpha=0.9, label='Derivada del EAR (Velocidad)')
        ax2.set_title(r"2. Velocidad de Cambio del EAR ($\Delta$EAR / $\Delta$t)", fontsize=12)
        ax2.set_ylabel('Velocidad', fontsize=10)
        ax2.grid(True, alpha=0.5, linestyle='--')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax2.legend(loc='upper right')

        # ------------------------------------------------
        # SUBPLOT 3: SEGUNDA DERIVADA (ACELERACIÓN)
        # ------------------------------------------------
        
        ax_accel.plot(timestamps, ear_accel, 'c-', linewidth=1.5, alpha=0.9, label='Aceleración del EAR')
        ax_accel.set_title(r'3. Aceleración del Cambio del EAR ($\Delta^2$EAR / $\Delta$t$^2$)', fontsize=12)
        ax_accel.set_ylabel('Aceleración', fontsize=10)
        ax_accel.grid(True, alpha=0.5, linestyle='--')
        ax_accel.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        # LÍNEA DE UMBRAL DE ACELERACIÓN (ÚNICO CRITERIO)
        ax_accel.axhline(y=accel_thresh, color='r', linestyle=':', linewidth=1, label=f'Umbral Único de Parpadeo (> {accel_thresh:.3f})')

        ax_accel.legend(loc='upper right')

        # ------------------------------------------------
        # SUBPLOT 4: MARCADO DE PARPADEOS DETECTADOS
        # ------------------------------------------------
        ax3.plot(timestamps, smoothed_ear, 'k-', linewidth=1.5, alpha=0.9, label='EAR Suavizado (SG)')
        
        # *** Título con corrección para MathText (sin cambios desde la última corrección) ***
        ax3.set_title(r'4. Detección Final de Parpadeo (Aceleración > {0:.3f} Y Duración $\geq$ {1} Frames) - Total: {2}'.format(
            accel_thresh, min_duration_frames, len(blinks)), fontsize=12)
        # *******************************************************************
        
        ax3.set_xlabel('Tiempo (segundos)', fontsize=12)
        ax3.set_ylabel('Valor EAR', fontsize=10)
        ax3.grid(True, alpha=0.5, linestyle='--')
        ax3.set_ylim(0, 0.6)
        
        for blink in blinks:
            ax3.axvspan(
                blink['start_time'], 
                blink['end_time'], 
                color='red', 
                alpha=0.3, 
                linewidth=0,
                label='Parpadeo Detectado' if blinks.index(blink) == 0 else ""
            )
        ax3.legend(loc='upper right')
    
        
        safe_name = video_filename.replace('.', '_')
        # Nuevo nombre de archivo para reflejar el criterio de duración de 2f
        output_path = os.path.join(output_dir, f"Resultados_EAR_{safe_name}.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig) 
        
        print(f"   🖼️ Gráfico de derivadas (EAR, Velocidad, Aceleración) guardado en: {output_path}")

    except Exception as e:
        # Imprimir el error para depuración
        print(f"   ⚠ Error al generar los gráficos para {video_filename}: {str(e)}")


# ------------------------------------------------------------------------------
# EJECUCIÓN DEL PROGRAMA
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # La carpeta de la persona que se va a analizar
    videos_path = r'C:\Users\Victor\Documents\Tesis2\Videos\Carlos'
    
    # *** CAMBIO: Directorio actualizado para reflejar la duración mínima de 2f ***
    results_path = r"C:\Users\Victor\Documents\Tesis2\Parpadeos\Resultados\Carlos"
    # **************************************************************************
    
    if not os.path.isdir(videos_path):
        print(f"Error: La ruta de videos '{videos_path}' no existe o no es un directorio.")
    else:
        process_videos_in_folder(videos_path, results_path)