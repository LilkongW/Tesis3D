import cv2
import numpy as np
import os
import time
import queue
import csv  # <--- IMPORTANTE: Necesario para guardar el log
from threading import Thread, Event, Lock
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN PRINCIPAL
# ============================================================================

# Índice de la webcam
WEBCAM_INDEX = 1

# Experimentos disponibles:
# 0 = Punto fijo central (30 segundos)
# 1 = 9 Puntos (1 ciclo)
# 2 = 5 Puntos (2 ciclos) - Calibración
# 3 = Espiral elíptica suave
EXP_NUM = 3

# CONFIGURACIÓN DE ALTA VELOCIDAD (120 FPS)
TARGET_CAM_WIDTH = 320   # QVGA - único modo con 120 FPS
TARGET_CAM_HEIGHT = 240  # QVGA
TARGET_CAM_FPS = 120.0   # FPS de captura
OUTPUT_FPS = 60.0        # FPS del video de salida (más estable para archivos)

# Ruta base para guardar videos
BASE_SAVE_PATH = "/home/vit/Documentos/Tesis3D/Videos"

# ============================================================================
# CONFIGURACIÓN DE PANTALLA Y VISUAL (VERSIÓN CORREGIDA)
# ============================================================================

# Obtener resolución de pantalla automáticamente
import tkinter as tk

def get_screen_resolution():
    """Obtiene la resolución de pantalla de forma segura"""
    try:
        root = tk.Tk()
        root.update_idletasks()  # Forzar actualización
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception as e:
        print(f"[PANTALLA] ⚠️ Error al detectar resolución: {e}")
        print("[PANTALLA] Usando resolución por defecto: 1920x1080")
        return 1920, 1080

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_resolution()
print(f"[PANTALLA] Resolución detectada: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Configuración del círculo
CIRCLE_RADIUS = 30
CIRCLE_COLOR = (0, 0, 255)  # Rojo en BGR
BG_COLOR = (0, 0, 0)  # Negro

# Configuración de cuadrícula 3x3
GRID_ROWS, GRID_COLS = 3, 3
CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================

frame_queue = queue.Queue(maxsize=240)  # Mayor capacidad para 120 FPS
cam_height, cam_width = TARGET_CAM_HEIGHT, TARGET_CAM_WIDTH
recording_active = False
capture_active = True
start_recording = Event()
recording_stopped = Event()
current_video_writer = None
current_output_path = None
writer_lock = Lock()

# Variables para sincronización
experiment_frame_times = []
webcam_frame_times = []

# ============================================================================
# FUNCIONES DE CAPTURA DE WEBCAM (OPTIMIZADA PARA 120 FPS)
# ============================================================================

def capture_webcam_stream():
    """Hilo de captura continua de la webcam a 120 FPS"""
    global cam_width, cam_height, capture_active, frame_queue, webcam_frame_times
    
    print(f"[WEBCAM] Iniciando captura a 120 FPS (índice {WEBCAM_INDEX})...")
    
    # Forzar el backend V4L2 de Linux
    cap_webcam = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_V4L2)
    
    if not cap_webcam.isOpened():
        print("[WEBCAM] ❌ Error: No se pudo abrir la webcam.")
        capture_active = False
        return
    
    # Forzar el formato MJPG (Motion-JPEG) para 120 FPS
    try:
        fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        cap_webcam.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
        print("[WEBCAM] ✓ Codec MJPG establecido")
    except Exception as e:
        print(f"[WEBCAM] ⚠️ Error al establecer MJPG: {e}")
    
    # Configurar webcam a QVGA @ 120 FPS
    cap_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_CAM_WIDTH)
    cap_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_CAM_HEIGHT)
    cap_webcam.set(cv2.CAP_PROP_FPS, TARGET_CAM_FPS)
    
    # Verificar configuración real
    actual_fps = cap_webcam.get(cv2.CAP_PROP_FPS)
    actual_width = int(cap_webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap_webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[WEBCAM] Solicitado: {TARGET_CAM_WIDTH}x{TARGET_CAM_HEIGHT} @ {TARGET_CAM_FPS} FPS")
    print(f"[WEBCAM] ✓ Configuración real: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    if actual_fps < 100:
        print(f"[WEBCAM] ⚠️ ADVERTENCIA: FPS más bajos de lo esperado ({actual_fps})")
    
    # Obtener primer frame
    ret, first_frame = cap_webcam.read()
    if ret:
        cam_height, cam_width, _ = first_frame.shape
        print(f"[WEBCAM] ✓ Resolución capturada: {cam_width}x{cam_height}")
        if not frame_queue.full():
            frame_queue.put((first_frame, time.time()))
    else:
        print("[WEBCAM] ❌ No se pudo leer el primer frame.")
        cap_webcam.release()
        capture_active = False
        return
    
    frame_count = 0
    last_fps_print = time.time()
    fps_measure_start = time.time()
    
    print("[WEBCAM] 🚀 Captura a alta velocidad iniciada...")
    
    while capture_active:
        ret, frame = cap_webcam.read()
        if not ret:
            print("[WEBCAM] ❌ Error al leer frame")
            break
        
        timestamp = time.time()
        
        # Mantener cola actualizada (más agresivo para 120 FPS)
        while frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        
        try:
            frame_queue.put((frame, timestamp), block=False)
        except queue.Full:
            pass  # Descartar frame si la cola está llena
        
        # Guardar timestamp si estamos grabando
        if recording_active:
            webcam_frame_times.append(timestamp)
        
        frame_count += 1
        
        # Estadísticas cada 5 segundos
        current_time = time.time()
        if current_time - last_fps_print >= 5.0:
            elapsed = current_time - fps_measure_start
            fps = frame_count / elapsed
            print(f"[WEBCAM] FPS real: {fps:.2f} | Cola: {frame_queue.qsize()} | Frames: {frame_count}")
            last_fps_print = current_time
    
    cap_webcam.release()
    total_time = time.time() - fps_measure_start
    final_fps = frame_count / total_time if total_time > 0 else 0
    print(f"[WEBCAM] Captura detenida. FPS promedio: {final_fps:.2f}")


def recording_worker():
    """Worker que graba frames de la webcam"""
    global recording_active, capture_active, frame_queue
    global current_video_writer, current_output_path
    
    print("[RECORDER] Worker iniciado...")
    
    while capture_active:
        start_recording.wait()
        
        if not capture_active:
            break
        
        with writer_lock:
            if current_video_writer is None or not current_video_writer.isOpened():
                print("[RECORDER] ❌ No hay VideoWriter válido")
                start_recording.clear()
                continue
        
        print(f"[RECORDER] 🔴 GRABACIÓN INICIADA: {current_output_path}")
        recording_stopped.clear()
        
        frame_count = 0
        dropped_frames = 0
        recording_start = time.time()
        last_status_print = time.time()
        
        while recording_active and capture_active:
            try:
                # Timeout más corto para 120 FPS
                frame, timestamp = frame_queue.get(timeout=0.1)
                
                with writer_lock:
                    if current_video_writer is not None:
                        current_video_writer.write(frame)
                        frame_count += 1
                
                # Estadísticas cada 3 segundos
                current_time = time.time()
                if current_time - last_status_print >= 3.0:
                    elapsed = current_time - recording_start
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"[RECORDER] FPS Escritura: {fps:.2f} | Frames: {frame_count} | Cola: {frame_queue.qsize()}")
                    last_status_print = current_time
                
            except queue.Empty:
                dropped_frames += 1
                if dropped_frames % 20 == 0:
                    print(f"[RECORDER] ⚠️ Frames perdidos (Buffer vacío): {dropped_frames}")
                if not capture_active:
                    break
                continue
        
        # Finalizar grabación
        duration = time.time() - recording_start
        print("[RECORDER] ✓ Grabación finalizada")
        print(f"[RECORDER]   Frames Guardados: {frame_count}")
        print(f"[RECORDER]   Duración Grabación: {duration:.2f}s")
        print(f"[RECORDER]   FPS Promedio Escritura: {frame_count/duration:.2f}")
        
        with writer_lock:
            if current_video_writer is not None:
                current_video_writer.release()
                current_video_writer = None
        
        recording_stopped.set()
        start_recording.clear()
    
    print("[RECORDER] Worker finalizado.")


def prepare_video_writer(output_path, width, height):
    """Prepara VideoWriter para grabación a 60 FPS"""
    global current_video_writer, current_output_path
    
    print(f"[SETUP] Preparando VideoWriter a {OUTPUT_FPS} FPS: {output_path}")
    
    # Usar mp4v para mejor compatibilidad
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (width, height))
    
    if not writer.isOpened():
        print("[SETUP] ⚠️ Intentando con XVID (.avi)")
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (width, height))
    
    if not writer.isOpened():
        print("[SETUP] ❌ Error: No se pudo inicializar VideoWriter")
        return False
    
    with writer_lock:
        current_video_writer = writer
        current_output_path = output_path
    
    print(f"[SETUP] ✓ VideoWriter listo (salida a {OUTPUT_FPS} FPS)")
    return True


def purge_frame_queue():
    """Limpia la cola de frames"""
    count = 0
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            count += 1
        except queue.Empty:
            break
    return count

# ============================================================================
# FUNCIONES AUXILIARES DE VISUALIZACIÓN
# ============================================================================

def show_countdown(window_name, text, duration_ms=2000):
    """Muestra un mensaje de cuenta regresiva"""
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text_size = cv2.getTextSize(text, font, 3, 8)[0]
    text_x = (SCREEN_WIDTH - text_size[0]) // 2
    text_y = (SCREEN_HEIGHT + text_size[1]) // 2
    
    cv2.putText(frame, text, (text_x, text_y), font, 3, (100, 200, 255), 8, cv2.LINE_AA)
    cv2.imshow(window_name, frame)
    cv2.waitKey(duration_ms)


def show_number_countdown(window_name):
    """Cuenta regresiva 3-2-1"""
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for count in range(3, 0, -1):
        frame.fill(0)
        text = str(count)
        text_size = cv2.getTextSize(text, font, 10, 25)[0]
        text_x = (SCREEN_WIDTH - text_size[0]) // 2
        text_y = (SCREEN_HEIGHT + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, 10, (255, 255, 255), 25, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1000)

# ============================================================================
# GENERADORES DE EXPERIMENTOS
# ============================================================================

def get_cell_center(row, col):
    """Calcula el centro de una celda en la cuadrícula"""
    x = col * CELL_WIDTH + CELL_WIDTH // 2
    y = row * CELL_HEIGHT + CELL_HEIGHT // 2
    return (x, y)


def generate_experiment_0():
    """Experimento 0: Punto fijo central por 30 segundos"""
    duration_seconds = 30
    fps = 60
    total_frames = duration_seconds * fps
    
    center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    positions = [center] * total_frames
    
    return positions, fps, "punto_fijo"


def generate_experiment_1():
    """Experimento 1: 9 puntos (1 ciclo)"""
    positions = []
    fps = 30
    frames_per_fixation = int(2 * fps)  # 2 segundos por punto
    
    # Generar 9 posiciones en orden de lectura
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            pos = get_cell_center(row, col)
            positions.extend([pos] * frames_per_fixation)
    
    return positions, fps, "9_puntos"


def generate_experiment_2():
    """Experimento 2: 5 puntos (2 ciclos)"""
    positions = []
    fps = 30
    frames_per_fixation = int(2 * fps)  # 2 segundos por punto
    
    # 5 posiciones: 4 esquinas + centro
    pos_sequence = [
        get_cell_center(0, 0),  # Superior izquierda
        get_cell_center(0, 2),  # Superior derecha
        get_cell_center(2, 0),  # Inferior izquierda
        get_cell_center(2, 2),  # Inferior derecha
        get_cell_center(1, 1),  # Centro
    ]
    
    # Repetir 2 veces
    for _ in range(2):
        for pos in pos_sequence:
            positions.extend([pos] * frames_per_fixation)
    
    return positions, fps, "5_puntos"


def generate_experiment_3():
    """Experimento 3: Espiral elíptica suave"""
    # Parámetros de la espiral
    k = 1.0
    theta_max = 15 * np.pi
    velocity = 230
    
    CENTER_X = SCREEN_WIDTH // 2
    CENTER_Y = SCREEN_HEIGHT // 2
    
    max_r = k * theta_max
    A = (SCREEN_WIDTH / 2) / max_r * 0.95
    B = (SCREEN_HEIGHT / 2) / max_r * 0.95
    
    # Calcular trayectoria
    num_points = 5000
    theta_values = np.linspace(0, theta_max, num_points)
    r_values = k * theta_values
    
    x_values = CENTER_X + A * r_values * np.cos(theta_values)
    y_values = CENTER_Y + B * r_values * np.sin(theta_values)
    
    # Calcular longitud de arco
    arc_length = np.zeros(len(x_values))
    for i in range(1, len(x_values)):
        dx = x_values[i] - x_values[i - 1]
        dy = y_values[i] - y_values[i - 1]
        arc_length[i] = arc_length[i - 1] + np.sqrt(dx**2 + dy**2)
    
    total_length = arc_length[-1]
    time_values = arc_length / total_length
    
    # Interpolar a velocidad uniforme
    fps = 60
    total_time = total_length / velocity
    num_frames = int(total_time * fps)
    
    interp_time = np.linspace(0, 1, num_frames)
    interp_x = np.interp(interp_time, time_values, x_values)
    interp_y = np.interp(interp_time, time_values, y_values)
    
    # Invertir dirección (de afuera hacia adentro)
    interp_x = interp_x[::-1]
    interp_y = interp_y[::-1]
    
    # Convertir a lista de posiciones
    positions = [(int(x), int(y)) for x, y in zip(interp_x, interp_y)]
    
    return positions, fps, "espiral"

# ============================================================================
# FUNCIÓN PRINCIPAL DE EXPERIMENTO (MODIFICADA PARA LOGUEO)
# ============================================================================

def run_experiment_iteration(positions, fps, exp_name, nombre_persona, numero_intento, save_path):
    """Ejecuta una iteración del experimento y guarda LOG de estímulos"""
    global recording_active, experiment_frame_times, webcam_frame_times
    
    # Resetear timestamps
    experiment_frame_times = []
    webcam_frame_times = []
    
    # Preparar rutas de archivos
    output_filename_video = f"{nombre_persona}_{exp_name}_intento_{numero_intento}.mp4"
    output_video_path = os.path.join(save_path, output_filename_video)
    
    output_filename_csv = f"{nombre_persona}_{exp_name}_intento_{numero_intento}_stimulus.csv"
    output_csv_path = os.path.join(save_path, output_filename_csv)
    
    if not prepare_video_writer(output_video_path, cam_width, cam_height):
        print(f"[{exp_name}] ❌ Error al preparar grabación de video")
        return False
    
    # Cuenta regresiva
    show_number_countdown("Experiment")
    show_countdown("Experiment", "Comenzando...", 1000)
    
    # Purgar cola
    purged = purge_frame_queue()
    print(f"[{exp_name}] Cola purgada: {purged} frames")
    time.sleep(0.15)
    
    print(f"\n{'⚡'*35}")
    print(f"   INICIANDO {exp_name.upper()} (Intento {numero_intento})")
    print(f"   LOG: {output_filename_csv}")
    print(f"{'⚡'*35}\n")
    
    # Abrir el archivo CSV para guardar el registro de estímulos
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            # --- CABECERA DEL LOG ---
            # timestamp_unix: Hora exacta del sistema (para cruzar con el ojo)
            # relative_time_s: Tiempo desde que inició el experimento
            # stimulus_x/y: Coordenada del punto rojo
            # frame_idx: Número de frame mostrado
            log_writer.writerow(["timestamp_unix", "relative_time_s", "stimulus_x", "stimulus_y", "frame_idx"])
            
            # Activar grabación
            recording_active = True
            start_recording.set()
            time.sleep(0.05)
            
            # Renderizar experimento frame por frame
            window_name = "Experiment"
            frame_interval = 1.0 / fps
            next_frame_time = time.time()
            
            experiment_start = time.time()
            
            for idx, pos in enumerate(positions):
                current_time = time.time()
                
                # Control de timing preciso
                if current_time < next_frame_time:
                    delay_ms = max(1, int((next_frame_time - current_time) * 1000))
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key == ord('q'):
                        recording_active = False
                        return False
                    # Recapturamos el tiempo justo antes de dibujar/guardar para máxima precisión
                    current_time = time.time()
                
                # Crear frame del experimento
                frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                cv2.circle(frame, pos, CIRCLE_RADIUS, CIRCLE_COLOR, -1)
                
                # Mostrar frame
                cv2.imshow(window_name, frame)
                
                # --- LOGGING: Guardar datos del estímulo ---
                elapsed_time = current_time - experiment_start
                log_writer.writerow([f"{current_time:.4f}", f"{elapsed_time:.4f}", pos[0], pos[1], idx])
                # -------------------------------------------
                
                # Guardar timestamp para sync global (opcional, ya lo tenemos en CSV)
                experiment_frame_times.append(current_time)
                
                # Actualizar timing
                next_frame_time += frame_interval
                
                # Progreso cada 100 frames
                if (idx + 1) % 100 == 0:
                    progress = (idx + 1) / len(positions) * 100
                    print(f"[{exp_name}] Progreso: {progress:.1f}% | Tiempo: {elapsed_time:.1f}s")
                
                # Check rápido de salida
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    recording_active = False
                    return False
            
    except IOError as e:
        print(f"❌ Error al escribir el archivo de Log CSV: {e}")
        recording_active = False
        return False
    
    # Finalizar
    experiment_duration = time.time() - experiment_start
    print(f"\n[{exp_name}] ✓ Experimento completado")
    print(f"[{exp_name}] Duración: {experiment_duration:.2f}s")
    print(f"[{exp_name}] Frames mostrados: {len(positions)}")
    print(f"[{exp_name}] Log guardado exitosamente.")
    
    # Detener grabación de video
    recording_active = False
    if not recording_stopped.wait(timeout=3.0):
        print(f"[{exp_name}] ⚠️ Timeout esperando finalización del video")
    
    # Análisis de sincronización rápido en consola
    if experiment_frame_times and webcam_frame_times:
        exp_start = experiment_frame_times[0]
        web_start = webcam_frame_times[0]
        offset = abs(exp_start - web_start)
        print("\n[SYNC] ╔═══════════════════════════════════")
        print(f"[SYNC] Offset inicial (Video vs Estímulo): {offset*1000:.2f} ms")
        print(f"[SYNC] Frames experimento: {len(experiment_frame_times)}")
        print(f"[SYNC] Frames webcam grabados: {len(webcam_frame_times)}")
        print("[SYNC] ╚═══════════════════════════════════\n")
    
    print(f"[{exp_name}] ✓ Iteración Completada\n")
    return True

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def run_all_experiments(nombre_persona, total_iteraciones, exp_num):
    """Ejecuta todas las iteraciones del experimento seleccionado"""
    global capture_active, recording_active
    
    print("\n" + "="*70)
    print("   SISTEMA DE EYE TRACKING @ 120 FPS + DATA LOGGING")
    print("="*70)
    
    # Crear carpeta de guardado
    save_path = os.path.join(BASE_SAVE_PATH, f"Experimento_{exp_num}", nombre_persona)
    os.makedirs(save_path, exist_ok=True)
    print(f"[SETUP] Carpeta: {save_path}")
    
    # Generar experimento
    print(f"[SETUP] Generando experimento {exp_num}...")
    
    if exp_num == 0:
        positions, fps, exp_name = generate_experiment_0()
    elif exp_num == 1:
        positions, fps, exp_name = generate_experiment_1()
    elif exp_num == 2:
        positions, fps, exp_name = generate_experiment_2()
    elif exp_num == 3:
        positions, fps, exp_name = generate_experiment_3()
    else:
        print(f"❌ Experimento {exp_num} no válido")
        return
    
    print(f"[SETUP] ✓ Experimento: {exp_name}")
    print(f"[SETUP] ✓ FPS experimento: {fps}")
    print(f"[SETUP] ✓ Total frames: {len(positions)}")
    
    # Inicializar captura
    print("\n" + "-"*70)
    print("   INICIALIZANDO SISTEMA @ 120 FPS")
    print("-"*70)
    
    capture_thread = Thread(target=capture_webcam_stream, daemon=True)
    capture_thread.start()
    time.sleep(2.5)  # Estabilización
    
    if not capture_active:
        print("❌ Error: Webcam no disponible")
        return
    
    recording_thread = Thread(target=recording_worker, daemon=True)
    recording_thread.start()
    
    print(f"[SISTEMA] ✓ Webcam: {cam_width}x{cam_height} @ 120 FPS (captura)")
    print(f"[SISTEMA] ✓ Grabación: {OUTPUT_FPS} FPS (video)")
    print("[SISTEMA] ✓ Sistema listo\n")
    
    # Crear ventana fullscreen
    cv2.namedWindow("Experiment", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Experiment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Ejecutar iteraciones
    for i in range(1, total_iteraciones + 1):
        if i > 1:
            show_countdown("Experiment", f"Preparate - Intento {i}", 3000)
        
        success = run_experiment_iteration(
            positions=positions,
            fps=fps,
            exp_name=exp_name,
            nombre_persona=nombre_persona,
            numero_intento=str(i),
            save_path=save_path
        )
        
        if not success:
            print(f"\n⚠️ Sesión interrumpida en intento {i}")
            break
        
        print(f"\n--- Fin del Intento {i} ---\n")
    
    # Limpieza
    print("\n" + "="*70)
    print("   FINALIZANDO SESIÓN")
    print("="*70)
    
    recording_active = False
    capture_active = False
    start_recording.set()
    
    capture_thread.join(timeout=3.0)
    recording_thread.join(timeout=3.0)
    
    cv2.destroyAllWindows()
    
    print("\n" + "✅"*35)
    print("   SESIÓN COMPLETADA")
    print("✅"*35 + "\n")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   SISTEMA DE EYE TRACKING @ 120 FPS v3.2 (Con Logging)")
    print("="*70)
    print("\nCONFIGURACIÓN DE ALTA VELOCIDAD:")
    print(f"  • Captura: {TARGET_CAM_WIDTH}x{TARGET_CAM_HEIGHT} @ {TARGET_CAM_FPS} FPS (MJPG)")
    print(f"  • Salida:  {OUTPUT_FPS} FPS (video)")
    print("\nExperimentos disponibles:")
    print("  0 = Punto fijo central (30 segundos)")
    print("  1 = 9 Puntos (1 ciclo, 2s por punto)")
    print("  2 = 5 Puntos (2 ciclos, 2s por punto) - Calibración")
    print("  3 = Espiral elíptica suave")
    print()
    
    nombre_persona = input("Nombre de la persona: ").strip()
    while not nombre_persona:
        print("❌ El nombre no puede estar vacío")
        nombre_persona = input("Nombre de la persona: ").strip()
    
    exp_input = input(f"Número de experimento (0-3, default={EXP_NUM}): ").strip()
    if exp_input:
        exp_num = int(exp_input)
        if exp_num not in [0, 1, 2, 3]:
            print(f"❌ Experimento inválido. Usando {EXP_NUM}")
            exp_num = EXP_NUM
    else:
        exp_num = EXP_NUM
    
    iteraciones_input = input("Número de iteraciones (default=1): ").strip()
    total_iteraciones = int(iteraciones_input) if iteraciones_input.isdigit() else 1
    
    print("\n" + "-"*70)
    print(f"✓ Participante: {nombre_persona}")
    print(f"✓ Experimento:  {exp_num}")
    print(f"✓ Iteraciones:  {total_iteraciones}")
    print("✓ Logs:         ACTIVADO (.csv)")
    print("-"*70)
    
    input("\nPresiona ENTER para comenzar...")
    
    run_all_experiments(nombre_persona, total_iteraciones, exp_num)
    
    print("\n🎉 Sistema finalizado. ¡Gracias!\n")