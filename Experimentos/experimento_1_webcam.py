import cv2
import os
import time
import numpy as np
import queue
from threading import Thread, Event

# Índice de la webcam
WEBCAM_INDEX = 1    

# Cola para almacenar frames de la Webcam
frame_queue = queue.Queue(maxsize=60)
screen_width, screen_height = 1920, 1080

# Variables globales para las dimensiones
cam_height, cam_width = 480, 640

# Variables de control
recording_active = False
capture_active = True

# Eventos para sincronización
videowriter_ready = Event()  # Indica que el VideoWriter está listo
start_recording = Event()    # Señal para iniciar grabación sincronizada

# Función para capturar frames de la WEBCAM
def capture_webcam_stream():
    global cam_width, cam_height, capture_active, frame_queue
    
    print(f"Iniciando captura de Webcam (índice {WEBCAM_INDEX})...")
    cap_webcam = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap_webcam.isOpened():
        print(f"Error: No se pudo abrir la Webcam (índice {WEBCAM_INDEX}).")
        capture_active = False
        return
    
    # Obtener las dimensiones del stream
    ret, first_frame = cap_webcam.read()
    if ret:
        cam_height, cam_width, _ = first_frame.shape
        print(f"Resolución de la Webcam detectada: {cam_width}x{cam_height}")
        
        if not frame_queue.full():
            frame_queue.put(first_frame)
    else:
        print("Advertencia: No se pudo leer el primer frame de la Webcam.")
        cap_webcam.release()
        return
    
    frame_count = 0
    last_fps_print_time = time.time()
    
    while capture_active:
        ret, frame = cap_webcam.read()
        if not ret:
            print("Error al leer frame de la Webcam. Deteniendo captura.")
            break
        
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        frame_queue.put(frame)
        frame_count += 1
        
        current_time = time.time()
        if current_time - last_fps_print_time >= 5:
            fps = frame_count / (current_time - last_fps_print_time)
            print(f"Webcam captura FPS: {fps:.2f}")
            frame_count = 0
            last_fps_print_time = current_time
    
    cap_webcam.release()
    print("Captura de Webcam detenida.")


def record_webcam_frames(output_video_path, rec_width, rec_height):
    """
    Función optimizada de grabación:
    1. Pre-inicializa el VideoWriter
    2. Señala que está lista
    3. Espera señal de inicio
    4. Comienza a grabar inmediatamente
    """
    global recording_active, capture_active, frame_queue
    
    try:
        print(f"[RECORDER] Pre-inicializando VideoWriter con {rec_width}x{rec_height}...")
        
        # PASO 1: Inicializar el VideoWriter (esto toma ~1 segundo)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (rec_width, rec_height))
        
        if not out.isOpened():
            print("[RECORDER] Advertencia: Falló 'mp4v'. Intentando con 'XVID' (.avi)")
            output_video_path = output_video_path.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (rec_width, rec_height))
        
        if not out.isOpened():
            print(f"[RECORDER] Error fatal: No se pudo abrir el VideoWriter")
            videowriter_ready.set()  # Señalar aunque haya error para no bloquear
            return
        
        # PASO 2: Señalar que estamos listos
        print("[RECORDER] ✓ VideoWriter inicializado. LISTO para grabar.")
        videowriter_ready.set()
        
        # PASO 3: Esperar señal de inicio sincronizada
        print("[RECORDER] Esperando señal de inicio...")
        start_recording.wait()
        
        if not capture_active:
            print("[RECORDER] Grabación cancelada antes de iniciar.")
            out.release()
            return
        
        print(f"[RECORDER] ¡SEÑAL RECIBIDA! ¡GRABACIÓN INICIADA en {output_video_path}!")
        
        frame_count = 0
        last_fps_print_time = time.time()
        recording_start_time = time.time()
        
        # PASO 4: Grabar frames mientras recording_active sea True
        while recording_active:
            try:
                frame = frame_queue.get(timeout=0.5)
                out.write(frame)
                frame_count += 1
                
                current_time = time.time()
                if current_time - last_fps_print_time >= 5:
                    fps = frame_count / (current_time - last_fps_print_time)
                    elapsed = current_time - recording_start_time
                    print(f"[RECORDER] FPS: {fps:.2f} | Frames: {frame_count} | Tiempo: {elapsed:.2f}s")
                    last_fps_print_time = current_time
                
            except queue.Empty:
                if not capture_active:
                    print("[RECORDER] La captura se detuvo. Finalizando grabación.")
                    break
                continue
        
        recording_duration = time.time() - recording_start_time
        out.release()
        print(f"[RECORDER] ✓ Video guardado: {output_video_path}")
        print(f"[RECORDER] Frames totales: {frame_count} | Duración: {recording_duration:.2f}s")
    
    except Exception as e:
        print(f"[RECORDER] Error: {e}")
        if 'out' in locals() and out.isOpened():
            out.release()


def run_experiment(nombre_persona, numero_intento):
    global recording_active, capture_active, frame_queue, cam_width, cam_height
    
    print("\n" + "="*70)
    print(f"EXPERIMENTO: {nombre_persona} - Intento {numero_intento}")
    print("="*70)

    # Resetear variables de control
    recording_active = False
    capture_active = True
    videowriter_ready.clear()
    start_recording.clear()
    
    # Limpiar la cola de frames
    print("[MAIN] Limpiando cola de frames...")
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    
    # Rutas
    Save_video_path = f"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/{nombre_persona}"
    experiment_video_path = "/home/vit/Documentos/Tesis3D/Videos/Animaciones_experimentos/experimento_1.mp4"

    if not os.path.exists(Save_video_path):
        os.makedirs(Save_video_path)

    cap_experiment = cv2.VideoCapture(experiment_video_path)
    
    if not cap_experiment.isOpened():
        print(f"[MAIN] Error: No se pudo abrir el video del experimento")
        capture_active = False
        return
    
    exp_width = int(cap_experiment.get(cv2.CAP_PROP_FRAME_WIDTH))
    exp_height = int(cap_experiment.get(cv2.CAP_PROP_FRAME_HEIGHT))
    exp_fps = cap_experiment.get(cv2.CAP_PROP_FPS)
    if exp_fps <= 0:
        exp_fps = 30.0
    
    print(f"[MAIN] Video experimento: {exp_width}x{exp_height} @ {exp_fps} FPS")
    
    # ==================================================================
    # FASE 1: INICIALIZACIÓN (durante cuenta regresiva)
    # ==================================================================
    
    # 1. Iniciar captura de webcam
    print("[MAIN] → Iniciando hilo de captura...")
    capture_thread = Thread(target=capture_webcam_stream, daemon=True)
    capture_thread.start()
    
    # 2. Esperar a que la webcam obtenga dimensiones
    print("[MAIN] → Esperando inicialización de webcam...")
    time.sleep(1.5)
    
    if not capture_active:
        print("[MAIN] ERROR: La webcam no pudo iniciarse. Abortando.")
        return
    
    print(f"[MAIN] ✓ Webcam lista: {cam_width}x{cam_height}")
    
    # 3. Iniciar hilo de grabación (pre-inicializa VideoWriter)
    output_filename = f"{nombre_persona}_intento_{numero_intento}.mp4"
    output_video_path = os.path.join(Save_video_path, output_filename)
    
    print("[MAIN] → Iniciando pre-inicialización del grabador...")
    recording_thread = Thread(
        target=record_webcam_frames, 
        args=(output_video_path, cam_width, cam_height), 
        daemon=True
    )
    recording_thread.start()
    
    # ==================================================================
    # FASE 2: CUENTA REGRESIVA (mientras VideoWriter se inicializa)
    # ==================================================================
    
    cv2.namedWindow("Experiment Video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Experiment Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("[MAIN] → Mostrando cuenta regresiva (VideoWriter inicializándose en paralelo)...")
    countdown_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    for count in range(3, 0, -1):
        countdown_frame.fill(0)
        text = str(count)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 10
        font_thickness = 25
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (screen_width - text_size[0]) // 2
        text_y = (screen_height + text_size[1]) // 2
        
        cv2.putText(countdown_frame, text, (text_x, text_y), font, 
                   font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.imshow("Experiment Video", countdown_frame)
        cv2.waitKey(1000)
    
    countdown_frame.fill(0)
    text = "Comenzando..."
    font_scale = 5
    font_thickness = 10
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (screen_width - text_size[0]) // 2
    text_y = (screen_height + text_size[1]) // 2
    cv2.putText(countdown_frame, text, (text_x, text_y), font, 
               font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.imshow("Experiment Video", countdown_frame)
    cv2.waitKey(1000)
    
    # ==================================================================
    # FASE 3: VERIFICAR QUE TODO ESTÉ LISTO
    # ==================================================================
    
    print("[MAIN] → Verificando que el VideoWriter esté listo...")
    if not videowriter_ready.wait(timeout=3.0):
        print("[MAIN] ERROR: VideoWriter no se inicializó a tiempo. Abortando.")
        capture_active = False
        return
    
    print("[MAIN] ✓ VideoWriter confirmado listo")
    
    # ==================================================================
    # FASE 4: PURGAR COLA Y SINCRONIZAR
    # ==================================================================
    
    print("[MAIN] → Purgando frames antiguos de la cuenta regresiva...")
    purged_count = 0
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            purged_count += 1
        except queue.Empty:
            break
    print(f"[MAIN] ✓ Cola purgada ({purged_count} frames eliminados)")
    
    # ==================================================================
    # FASE 5: REPRODUCIR EXPERIMENTO Y GRABAR SINCRONIZADAMENTE
    # ==================================================================
    
    print("\n" + "🔴"*35)
    print("INICIANDO EXPERIMENTO Y GRABACIÓN SINCRONIZADOS")
    print("🔴"*35 + "\n")
    
    next_frame_time = time.time()
    first_frame_shown = False
    
    while cap_experiment.isOpened():
        current_time = time.time()
        if current_time < next_frame_time:
            delay_ms = int((next_frame_time - current_time) * 1000)
            if delay_ms > 0:
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q'):
                    print("[MAIN] Experimento detenido manualmente.")
                    break
                continue
        
        next_frame_time = current_time + (1.0 / exp_fps)
        
        ret_exp, frame_exp = cap_experiment.read()
        if not ret_exp:
            print(f"[MAIN] Video del experimento finalizado.")
            break
        
        # ¡SINCRONIZACIÓN PERFECTA!
        # En el PRIMER frame del experimento, activamos la grabación
        if not first_frame_shown:
            print("\n" + "⚡"*35)
            print("¡SEÑAL DE INICIO SINCRONIZADO!")
            print("⚡"*35 + "\n")
            
            # Activar ambas señales simultáneamente
            recording_active = True
            start_recording.set()  # Libera al recorder instantáneamente
            first_frame_shown = True
        
        # Mostrar el video del experimento
        aspect_ratio = exp_width / exp_height
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
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"[MAIN] Experimento detenido manualmente (Intento {numero_intento}).")
            break
    
    # ==================================================================
    # FASE 6: DETENER GRABACIÓN
    # ==================================================================
    
    print("\n" + "🛑"*35)
    print(f"DETENIENDO GRABACIÓN (Intento {numero_intento})")
    print("🛑"*35 + "\n")
    
    recording_active = False
    capture_active = False
    
    print("[MAIN] Esperando finalización de hilos...")
    capture_thread.join(timeout=2.0)
    recording_thread.join(timeout=5.0)
    
    cap_experiment.release()
    cv2.destroyAllWindows()
    
    print(f"[MAIN] ✓ Intento {numero_intento} finalizado.")
    print("[MAIN] Esperando 3 segundos antes de continuar...")
    time.sleep(3)


# ==================================================================
# BUCLE PRINCIPAL
# ==================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("CONFIGURACIÓN DE EXPERIMENTOS CON SINCRONIZACIÓN OPTIMIZADA")
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
    print(f"✓ Persona: {nombre_persona}")
    print(f"✓ Total de intentos: {total_iteraciones}")
    print("-"*70 + "\n")
    
    for i in range(1, total_iteraciones + 1):
        run_experiment(nombre_persona, str(i))
    
    print("\n" + "="*70)
    print("✓ TODAS LAS ITERACIONES COMPLETADAS")
    print("="*70 + "\n")