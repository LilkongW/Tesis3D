import cv2
import os
import time
import numpy as np
import queue
from threading import Thread, Event, Lock

# --- 1. CONFIGURACIÓN PRINCIPAL ---

# Índice de la webcam
WEBCAM_INDEX = 1 

# Número de experimento (para seleccionar video de estímulo)
# 1 = 9 Puntos
# 2 = 5 Puntos (usado como calibración)
# 3 = Espiral (ACTIVARÁ EL MODO DE CALIBRACIÓN PREVIA EN VIVO)
EXP_NUM = 3

# FPS de grabación de la webcam (debe coincidir con la config de la webcam)
OUTPUT_FPS = 30

# --- 2. CONFIGURACIÓN DE RUTAS ---
# !!! Modifica estas rutas para que coincidan con tu PC !!!

# Ruta base para GUARDAR los videos de la webcam
BASE_SAVE_PATH = "C:\\Users\\Victor\\Documents\\Tesis3D\\Data" 

# Ruta base donde ESTÁN los videos de estímulo (los .mp4 generados)
STIMULUS_VIDEO_PATH = "C:\\Users\\Victor\\Documents\\Tesis3D\\Videos\\Animaciones_experimentos"

# --- 3. VARIABLES GLOBALES (No tocar) ---
frame_queue = queue.Queue(maxsize=120)
screen_width, screen_height = 1920, 1080
cam_height, cam_width = 480, 640
recording_active = False
capture_active = True
start_recording = Event()
recording_stopped = Event()
current_video_writer = None
current_output_path = None
writer_lock = Lock()


def capture_webcam_stream():
    """Hilo de captura continua - se ejecuta durante TODA la sesión"""
    global cam_width, cam_height, capture_active, frame_queue
    
    print(f"[WEBCAM] Iniciando captura continua (índice {WEBCAM_INDEX})...")
    cap_webcam = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap_webcam.isOpened():
        print("[WEBCAM] ❌ Error: No se pudo abrir la webcam.")
        capture_active = False
        return
    
    # Configurar webcam a 30 FPS y resolución estándar
    print("[WEBCAM] Configurando webcam a 30 FPS y 640x480...")
    cap_webcam.set(cv2.CAP_PROP_FPS, 30)
    cap_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Verificar configuración
    actual_fps = cap_webcam.get(cv2.CAP_PROP_FPS)
    actual_width = int(cap_webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap_webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("[WEBCAM] Configuración aplicada:")
    print(f"[WEBCAM]   - FPS: {actual_fps}")
    print(f"[WEBCAM]   - Resolución: {actual_width}x{actual_height}")
    
    if actual_fps < 25:
        print(f"[WEBCAM] ⚠️  ADVERTENCIA: FPS muy bajos ({actual_fps})")
    
    # Obtener dimensiones reales
    ret, first_frame = cap_webcam.read()
    if ret:
        cam_height, cam_width, _ = first_frame.shape
        print(f"[WEBCAM] ✓ Resolución real capturada: {cam_width}x{cam_height}")
        if not frame_queue.full():
            frame_queue.put((first_frame, time.time()))
    else:
        print("[WEBCAM] ❌ No se pudo leer el primer frame.")
        cap_webcam.release()
        capture_active = False
        return
    
    frame_count = 0
    last_fps_print_time = time.time()
    
    while capture_active:
        ret, frame = cap_webcam.read()
        if not ret:
            print("[WEBCAM] ❌ Error al leer frame. Deteniendo captura.")
            break
        
        timestamp = time.time()
        
        # Mantener solo los frames más recientes
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        frame_queue.put((frame, timestamp))
        frame_count += 1
        
        current_time = time.time()
        if current_time - last_fps_print_time >= 10:
            fps = frame_count / (current_time - last_fps_print_time)
            print(f"[WEBCAM] FPS de captura: {fps:.2f} (Cola: {frame_queue.qsize()})")
            frame_count = 0
            last_fps_print_time = current_time
    
    cap_webcam.release()
    print("[WEBCAM] Captura detenida.")


def recording_worker():
    """
    Worker de grabación que se mantiene activo durante toda la sesión.
    Graba mientras recording_active == True
    """
    global recording_active, capture_active, frame_queue
    global current_video_writer, current_output_path
    
    print("[RECORDER] Worker iniciado - esperando trabajos...")
    
    while capture_active:
        # Esperar señal de inicio
        start_recording.wait()
        
        if not capture_active:
            break
        
        # Verificar que tenemos un VideoWriter válido
        with writer_lock:
            if current_video_writer is None or not current_video_writer.isOpened():
                print("[RECORDER] ❌ Error: No hay VideoWriter válido")
                start_recording.clear()
                continue
        
        print(f"[RECORDER] 🔴 GRABACIÓN INICIADA: {current_output_path}")
        recording_stopped.clear()
        
        frame_count = 0
        dropped_frames = 0
        recording_start_time = time.time()
        last_fps_print_time = time.time()
        last_frame_count = 0
        
        # Grabar mientras recording_active sea True
        while recording_active and capture_active:
            try:
                frame, timestamp = frame_queue.get(timeout=0.5)
                
                with writer_lock:
                    if current_video_writer is not None:
                        current_video_writer.write(frame)
                        frame_count += 1
                
                current_time = time.time()
                if current_time - last_fps_print_time >= 5:
                    frames_in_interval = frame_count - last_frame_count
                    fps = frames_in_interval / (current_time - last_fps_print_time)
                    elapsed = current_time - recording_start_time
                    queue_size = frame_queue.qsize()
                    print(f"[RECORDER] FPS: {fps:.2f} | Total frames: {frame_count} | Tiempo: {elapsed:.2f}s | Cola: {queue_size}")
                    last_fps_print_time = current_time
                    last_frame_count = frame_count
                
            except queue.Empty:
                dropped_frames += 1
                if dropped_frames % 10 == 0:
                    print(f"[RECORDER] ⚠️  Frames perdidos por cola vacía: {dropped_frames}")
                if not capture_active:
                    break
                continue
        
        # Finalizar grabación
        recording_duration = time.time() - recording_start_time
        expected_frames = int(recording_duration * OUTPUT_FPS)
        print("[RECORDER] ⏹️  Grabación detenida")
        print(f"[RECORDER] ✓ Frames grabados: {frame_count}")
        print(f"[RECORDER] ✓ Duración: {recording_duration:.2f}s")
        print(f"[RECORDER] ✓ FPS promedio real: {frame_count/recording_duration:.2f}")
        
        # Liberar el VideoWriter
        with writer_lock:
            if current_video_writer is not None:
                current_video_writer.release()
                current_video_writer = None
        
        # Señalizar que la grabación terminó
        recording_stopped.set()
        
        # Limpiar señal para próxima iteración
        start_recording.clear()
    
    print("[RECORDER] Worker finalizado.")


def prepare_video_writer(output_path, width, height):
    """Prepara un nuevo VideoWriter para la siguiente iteración"""
    global current_video_writer, current_output_path
    
    print(f"[SETUP] Preparando VideoWriter a {OUTPUT_FPS} FPS: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (width, height))
    
    if not writer.isOpened():
        print("[SETUP] ⚠️  Falló 'mp4v', intentando 'XVID' (.avi)")
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (width, height))
    
    if not writer.isOpened():
        print("[SETUP] ❌ Error fatal: No se pudo inicializar VideoWriter")
        return False
    
    with writer_lock:
        current_video_writer = writer
        current_output_path = output_path
    
    print(f"[SETUP] ✓ VideoWriter listo a {OUTPUT_FPS} FPS")
    return True


def show_countdown(window_name, countdown_text="Prepárate", wait_time=2000):
    """Muestra un mensaje de cuenta regresiva"""
    countdown_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 6
    font_thickness = 15
    
    text_size = cv2.getTextSize(countdown_text, font, font_scale, font_thickness)[0]
    text_x = (screen_width - text_size[0]) // 2
    text_y = (screen_height + text_size[1]) // 2
    
    cv2.putText(countdown_frame, countdown_text, (text_x, text_y), font, 
                font_scale, (100, 200, 255), font_thickness, cv2.LINE_AA)
    cv2.imshow(window_name, countdown_frame)
    cv2.waitKey(wait_time)


def show_number_countdown(window_name):
    """Muestra cuenta regresiva 3-2-1"""
    countdown_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 10
    font_thickness = 25
    
    for count in range(3, 0, -1):
        countdown_frame.fill(0)
        text = str(count)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (screen_width - text_size[0]) // 2
        text_y = (screen_height + text_size[1]) // 2
        
        cv2.putText(countdown_frame, text, (text_x, text_y), font, 
                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.imshow(window_name, countdown_frame)
        cv2.waitKey(1000)


def purge_frame_queue():
    """Limpia la cola de frames para sincronización perfecta"""
    purged_count = 0
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            purged_count += 1
        except queue.Empty:
            break
    return purged_count


def run_experiment_iteration(cap_stimulus, stim_fps, stim_width, stim_height, total_frames, 
                             stim_name, nombre_persona, numero_intento, save_path):
    """
    Ejecuta UNA iteración de UN estímulo (ya sea calibración o espiral).
    """
    global recording_active
    
    # Preparar VideoWriter para esta iteración
    output_filename = f"{nombre_persona}_{stim_name}_intento_{numero_intento}.mp4"
    output_video_path = os.path.join(save_path, output_filename)
    
    if not prepare_video_writer(output_video_path, cam_width, cam_height):
        print(f"[{stim_name} Intento {numero_intento}] ❌ Error al preparar grabación. Saltando iteración.")
        return False
    
    # --- Cuenta regresiva 3-2-1 ---
    show_number_countdown("Experiment Video")
    
    # "Comenzando..."
    countdown_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    font_thickness = 10
    text = "Comenzando..."
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (screen_width - text_size[0]) // 2
    text_y = (screen_height + text_size[1]) // 2
    cv2.putText(countdown_frame, text, (text_x, text_y), font, 
                font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.imshow("Experiment Video", countdown_frame)
    cv2.waitKey(1000)
    
    # CRÍTICO: Purgar cola JUSTO antes de empezar
    purged = purge_frame_queue()
    print(f"[{stim_name} Intento {numero_intento}] Cola purgada ({purged} frames)")
    
    # Pequeña espera para que la cola se estabilice
    time.sleep(0.1)
    
    # Reiniciar video al inicio
    cap_stimulus.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"\n{'⚡'*35}")
    print(f"   ¡INICIANDO {stim_name.upper()} (Intento {numero_intento})!")
    print(f"{'⚡'*35}\n")
    
    # SINCRONIZACIÓN MEJORADA: Activar grabación ANTES del loop
    recording_active = True
    start_recording.set()
    
    # Pequeño delay para asegurar que el worker está listo
    time.sleep(0.05)
    
    # Reproducir experimento
    next_frame_time = time.time()
    frame_interval = 1.0 / stim_fps
    
    experiment_start_time = time.time()
    experiment_frame_count = 0
    
    while True:
        current_time = time.time()
        
        # Control de timing más preciso
        if current_time < next_frame_time:
            delay_ms = max(1, int((next_frame_time - current_time) * 1000))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                print(f"[{stim_name} Intento {numero_intento}] Detenido manualmente.")
                recording_active = False
                return False
            continue
        
        # Leer frame del experimento
        ret_exp, frame_exp = cap_stimulus.read()
        if not ret_exp:
            experiment_duration = time.time() - experiment_start_time
            print(f"[{stim_name} Intento {numero_intento}] ✓ Video finalizado.")
            print(f"[EXPERIMENTO] Duración real: {experiment_duration:.2f}s")
            print(f"[EXPERIMENTO] Frames mostrados: {experiment_frame_count}/{total_frames}")
            print(f"[EXPERIMENTO] FPS promedio: {experiment_frame_count/experiment_duration:.2f}")
            break
        
        experiment_frame_count += 1
        
        # Mostrar frame del experimento (con escalado y centrado)
        aspect_ratio = stim_width / stim_height
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)
        
        if new_width > screen_width:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)
        
        resized_exp = cv2.resize(frame_exp, (new_width, new_height))
        display_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        y_offset = (screen_height - new_height) // 2
        x_offset = (screen_width - new_width) // 2
        display_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_exp
        
        cv2.imshow("Experiment Video", display_frame)
        
        # Actualizar próximo tiempo de frame
        next_frame_time += frame_interval
        
        # Check rápido de tecla
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"[{stim_name} Intento {numero_intento}] Detenido manually.")
            recording_active = False
            return False
    
    # Detener grabación INMEDIATAMENTE
    print(f"\n{'🛑'*35}")
    print(f"   FINALIZANDO {stim_name.upper()} (Intento {numero_intento})")
    print(f"{'🛑'*35}\n")
    
    recording_active = False
    
    # Esperar a que el worker termine (con timeout)
    if not recording_stopped.wait(timeout=2.0):
        print(f"[{stim_name} Intento {numero_intento}] ⚠️  Timeout esperando finalización de grabación")
    
    print(f"[{stim_name} Intento {numero_intento}] ✓ Completado\n")
    return True


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- ¡NUEVA FUNCIÓN! ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def run_live_calibration(window_name, save_path, nombre_persona, numero_intento):
    """
    Genera los 5 puntos de calibración en vivo usando OpenCV
    y graba la webcam al mismo tiempo.
    """
    global recording_active

    print(f"\n[{'C'*35}]")
    print(f"   INICIANDO CALIBRACIÓN EN VIVO (Intento {numero_intento})")
    print(f"[{'C'*35}]\n")

    # --- 1. Definir Puntos y Tiempos ---
    # Coordenadas "Ground Truth" (basadas en tu script Crear_animaciones.py)
    cell_width = screen_width // 3
    cell_height = screen_height // 3
    pos_top_left = (cell_width // 2, cell_height // 2) # (320, 180)
    pos_top_right = (2 * cell_width + cell_width // 2, cell_height // 2) # (1600, 180)
    pos_bottom_left = (cell_width // 2, 2 * cell_height + cell_height // 2) # (320, 900)
    pos_bottom_right = (2 * cell_width + cell_width // 2, 2 * cell_height + cell_height // 2) # (1600, 900)
    pos_center = (cell_width + cell_width // 2, cell_height + cell_height // 2) # (960, 540)
    
    # Secuencia de puntos (repetida dos veces)
    pixel_sequence = [
        pos_top_left, pos_top_right, pos_bottom_left, pos_bottom_right, pos_center,
        pos_top_left, pos_top_right, pos_bottom_left, pos_bottom_right, pos_center
    ]
    fixation_duration_ms = 2000 # 2 segundos por punto

    # --- 2. Preparar Grabación ---
    stim_name = "calibracion_en_vivo"
    output_filename = f"{nombre_persona}_{stim_name}_intento_{numero_intento}.mp4"
    output_video_path = os.path.join(save_path, output_filename)
    
    if not prepare_video_writer(output_video_path, cam_width, cam_height):
        print(f"[{stim_name} Intento {numero_intento}] ❌ Error al preparar grabación.")
        return False

    # --- 3. Cuenta regresiva ---
    if numero_intento == "1": # Mostrar solo la primera vez
        show_countdown("Experiment Video", "Paso 1: Calibracion", 3000)
    show_number_countdown("Experiment Video")
    
    # "Comenzando..."
    countdown_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    font_thickness = 10
    text = "Comenzando..."
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (screen_width - text_size[0]) // 2
    text_y = (screen_height + text_size[1]) // 2
    cv2.putText(countdown_frame, text, (text_x, text_y), font, 
                font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.imshow("Experiment Video", countdown_frame)
    cv2.waitKey(1000)

    # --- 4. Iniciar Grabación y Bucle ---
    purged = purge_frame_queue()
    print(f"[{stim_name} Intento {numero_intento}] Cola purgada ({purged} frames)")
    time.sleep(0.1)

    print(f"\n{'⚡'*35}")
    print(f"   ¡INICIANDO CALIBRACIÓN (Intento {numero_intento})!")
    print(f"{'⚡'*35}\n")
    
    recording_active = True
    start_recording.set()
    time.sleep(0.05)
    
    frame_base = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    for i, point_coords in enumerate(pixel_sequence):
        print(f"[CALIBRACIÓN] Mostrando punto {i+1}/{len(pixel_sequence)} en {point_coords}")
        frame_base.fill(0) # Fondo negro
        cv2.circle(frame_base, point_coords, 30, (0, 0, 255), -1) # Círculo rojo
        
        start_time = time.time()
        while (time.time() - start_time) < (fixation_duration_ms / 1000.0):
            cv2.imshow("Experiment Video", frame_base)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"[{stim_name} Intento {numero_intento}] Detenido manualmente.")
                recording_active = False # ¡Importante! Detener la grabación
                return False # Salir de la función
        
        if not recording_active: # Doble chequeo si 'q' fue presionada
            break
    
    # --- 5. Finalizar Grabación ---
    print(f"\n{'🛑'*35}")
    print(f"   FINALIZANDO CALIBRACIÓN (Intento {numero_intento})")
    print(f"{'🛑'*35}\n")
    
    recording_active = False
    if not recording_stopped.wait(timeout=2.0):
        print(f"[{stim_name} Intento {numero_intento}] ⚠️  Timeout esperando finalización")
    
    print(f"[{stim_name} Intento {numero_intento}] ✓ Completado\n")
    
    # Mostrar pantalla en negro
    frame_base.fill(0)
    cv2.imshow("Experiment Video", frame_base)
    cv2.waitKey(1000) # Pausa de 1s
    
    return True
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


def run_all_experiments(nombre_persona, total_iteraciones):
    """Función principal que ejecuta TODAS las iteraciones en un solo flujo"""
    global capture_active, recording_active
    
    print("\n" + "="*70)
    print("   INICIANDO SESIÓN DE EXPERIMENTOS")
    print("="*70)
    
    # Configurar ruta de guardado
    save_path = os.path.join(BASE_SAVE_PATH, f"Experimento_{EXP_NUM}", nombre_persona)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"[SESIÓN] Creada carpeta de guardado: {save_path}")
    
    # --- Cargar Video de Estímulo PRINCIPAL ---
    experiment_video_path = os.path.join(STIMULUS_VIDEO_PATH, f"experimento_{EXP_NUM}.mp4")
    cap_experiment = cv2.VideoCapture(experiment_video_path)
    if not cap_experiment.isOpened():
        print(f"❌ Error: No se pudo abrir {experiment_video_path}")
        return
    
    exp_width = int(cap_experiment.get(cv2.CAP_PROP_FRAME_WIDTH))
    exp_height = int(cap_experiment.get(cv2.CAP_PROP_FRAME_HEIGHT))
    exp_fps = cap_experiment.get(cv2.CAP_PROP_FPS)
    if exp_fps <= 0: exp_fps = 30.0
    total_frames = int(cap_experiment.get(cv2.CAP_PROP_FRAME_COUNT))
    exp_duration = total_frames / exp_fps
    
    print(f"[VIDEO] Experimento Principal: {experiment_video_path}")
    print(f"[VIDEO]   -> {exp_width}x{exp_height} @ {exp_fps} FPS, Duración: {exp_duration:.2f}s")
    
    # --- ¡MODIFICADO! Ya no se carga el video de calibración ---
    if EXP_NUM == 3:
        print("[SESIÓN] EXP_NUM=3 detectado. Se ejecutará CALIBRACIÓN EN VIVO antes de la espiral.")

    
    # ==================================================================
    # INICIALIZACIÓN ÚNICA (Webcam y Recorder)
    # ==================================================================
    
    print("\n" + "-"*70)
    print("   INICIALIZANDO SISTEMA")
    print("-"*70)
    
    # 1. Iniciar captura continua
    capture_thread = Thread(target=capture_webcam_stream, daemon=True)
    capture_thread.start()
    print("[SISTEMA] Esperando que la webcam se estabilice...")
    time.sleep(2.0) # Dar tiempo a la webcam para que inicie y se obtenga la resolución
    
    if not capture_active:
        print("❌ Error: La webcam no pudo iniciarse.")
        cap_experiment.release()
        return
    
    # 2. Iniciar worker de grabación
    recording_thread = Thread(target=recording_worker, daemon=True)
    recording_thread.start()
    
    print(f"[SISTEMA] ✓ Webcam activa: {cam_width}x{cam_height}")
    print("[SISTEMA] ✓ Worker de grabación activo")
    print(f"[SISTEMA] ✓ Grabación configurada a {OUTPUT_FPS} FPS")
    print("[SISTEMA] ✓ Todo listo para comenzar\n")
    
    # ==================================================================
    # VENTANA FULLSCREEN
    # ==================================================================
    
    cv2.namedWindow("Experiment Video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Experiment Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # ==================================================================
    # BUCLE DE ITERACIONES
    # ==================================================================
    
    for i in range(1, total_iteraciones + 1):
        
        # Mostrar "Prepárate" si no es la primera iteración
        if i > 1:
            show_countdown("Experiment Video", f"Prepárate - Intento {i}", 3000)

        # --- (A) PASO DE CALIBRACIÓN (Solo si es EXP_NUM 3) ---
        # --- ¡MODIFICADO! Llama a la nueva función en vivo ---
        if EXP_NUM == 3:
            success_cal = run_live_calibration(
                window_name="Experiment Video",
                save_path=save_path,
                nombre_persona=nombre_persona,
                numero_intento=str(i)
            )
            
            if not success_cal:
                print(f"\n⚠️  Sesión interrumpida en CALIBRACIÓN (intento {i})")
                break # Salir del bucle de iteraciones
            
            # Pausa breve
            show_countdown("Experiment Video", "Calibracion Completa", 1500)
        
        # --- (B) PASO DE EXPERIMENTO (Siempre se ejecuta) ---
        print("\n" + "*"*70)
        print(f"   INICIANDO EXPERIMENTO {EXP_NUM} (Intento {i}/{total_iteraciones})")
        print("*"*70)
        
        # Mostrar "Prepárate" específico
        if EXP_NUM == 3:
             show_countdown("Experiment Video", "Paso 2: Espiral", 3000)
        elif i == 1: # Si es el primer intento (y no es exp 3)
             show_countdown("Experiment Video", "Prepárate", 3000)

        
        success_exp = run_experiment_iteration(
            cap_stimulus=cap_experiment,
            stim_fps=exp_fps,
            stim_width=exp_width,
            stim_height=exp_height,
            total_frames=total_frames,
            stim_name=f"experimento_{EXP_NUM}", # <--- Nombre de archivo
            nombre_persona=nombre_persona,
            numero_intento=str(i),
            save_path=save_path
        )
        
        if not success_exp:
            print(f"\n⚠️  Sesión interrumpida en EXPERIMENTO (intento {i})")
            break # Salir del bucle de iteraciones
        
        print(f"\n--- Fin del Intento {i} ---")

    
    # ==================================================================
    # LIMPIEZA FINAL
    # ==================================================================
    
    print("\n" + "="*70)
    print("   FINALIZANDO SESIÓN")
    print("="*70)
    
    recording_active = False
    capture_active = False
    start_recording.set()   # Liberar worker si está esperando
    
    print("[CLEANUP] Esperando finalización de hilos...")
    capture_thread.join(timeout=2.0)
    recording_thread.join(timeout=3.0)
    
    cap_experiment.release()
    # Ya no hay cap_calibration que liberar
    cv2.destroyAllWindows()
    
    print("\n" + "✅"*35)
    print("   TODAS LAS ITERACIONES COMPLETADAS")
    print("✅"*35 + "\n")


# ==================================================================
# PUNTO DE ENTRADA
# ==================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   SISTEMA DE EXPERIMENTOS - (v2.1 CON CALIBRACIÓN EN VIVO)")
    print("="*70 + "\n")
    
    nombre_persona = input("Nombre de la persona: ").strip()
    while not nombre_persona:
        print("❌ El nombre no puede estar vacío.")
        nombre_persona = input("Nombre de la persona: ").strip()

    total_iteraciones_str = input("Número total de iteraciones: ").strip()
    while not total_iteraciones_str.isdigit() or int(total_iteraciones_str) <= 0:
        print("❌ Debe ser un número positivo.")
        total_iteraciones_str = input("Número total de iteraciones: ").strip()
    
    total_iteraciones = int(total_iteraciones_str)

    print("\n" + "-"*70)
    print(f"✓ Participante: {nombre_persona}")
    print(f"✓ Iteraciones:  {total_iteraciones}")
    print(f"✓ Experimento:  {EXP_NUM}")
    if EXP_NUM == 3:
        print("✓ Modo:         ¡CALIBRACIÓN EN VIVO + ESPIRAL activado!")
    print(f"✓ FPS Salida:   {OUTPUT_FPS}")
    print(f"✓ Guardar en:   {os.path.join(BASE_SAVE_PATH, f'Experimento_{EXP_NUM}', nombre_persona)}")
    print(f"✓ Leer desde:   {STIMULUS_VIDEO_PATH}")
    print("-"*70)
    
    input("\nPresiona ENTER para comenzar...")
    
    run_all_experiments(nombre_persona, total_iteraciones)
    
    print("\n🎉 Sesión finalizada. ¡Gracias!\n")