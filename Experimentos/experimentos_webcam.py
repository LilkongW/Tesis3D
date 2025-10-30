import cv2
import os
import time
import numpy as np
import queue
from threading import Thread, Event

# Índice de la webcam
WEBCAM_INDEX = 1    

# Número de experimento (para seleccionar video de estímulo)
EXP_NUM = 1

# Cola para almacenar frames de la Webcam
frame_queue = queue.Queue(maxsize=60)
screen_width, screen_height = 1920, 1080

# Variables globales para las dimensiones
cam_height, cam_width = 480, 640

# Variables de control
recording_active = False
capture_active = True

# Eventos para sincronización
videowriter_ready = Event()
start_recording = Event()

# Variable global para el VideoWriter actual
current_video_writer = None
current_output_path = None


def capture_webcam_stream():
    """Hilo de captura continua - se ejecuta durante TODA la sesión"""
    global cam_width, cam_height, capture_active, frame_queue
    
    print(f"[WEBCAM] Iniciando captura continua (índice {WEBCAM_INDEX})...")
    cap_webcam = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap_webcam.isOpened():
        print("[WEBCAM] ❌ Error: No se pudo abrir la webcam.")
        capture_active = False
        return
    
    # Obtener dimensiones
    ret, first_frame = cap_webcam.read()
    if ret:
        cam_height, cam_width, _ = first_frame.shape
        print(f"[WEBCAM] ✓ Resolución detectada: {cam_width}x{cam_height}")
        if not frame_queue.full():
            frame_queue.put(first_frame)
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
        
        # Mantener solo los frames más recientes
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        frame_queue.put(frame)
        frame_count += 1
        
        current_time = time.time()
        if current_time - last_fps_print_time >= 10:
            fps = frame_count / (current_time - last_fps_print_time)
            print(f"[WEBCAM] FPS de captura: {fps:.2f}")
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
        if current_video_writer is None or not current_video_writer.isOpened():
            print("[RECORDER] ❌ Error: No hay VideoWriter válido")
            start_recording.clear()
            continue
        
        print(f"[RECORDER] 🔴 GRABACIÓN INICIADA: {current_output_path}")
        
        frame_count = 0
        recording_start_time = time.time()
        last_fps_print_time = time.time()
        
        # Grabar mientras recording_active sea True
        while recording_active and capture_active:
            try:
                frame = frame_queue.get(timeout=0.5)
                current_video_writer.write(frame)
                frame_count += 1
                
                current_time = time.time()
                if current_time - last_fps_print_time >= 10:
                    fps = frame_count / (current_time - last_fps_print_time)
                    elapsed = current_time - recording_start_time
                    print(f"[RECORDER] FPS: {fps:.2f} | Frames: {frame_count} | Tiempo: {elapsed:.2f}s")
                    last_fps_print_time = current_time
                
            except queue.Empty:
                if not capture_active:
                    break
                continue
        
        # Finalizar grabación
        recording_duration = time.time() - recording_start_time
        print("[RECORDER] ⏹️  Grabación detenida")
        print(f"[RECORDER] ✓ Frames: {frame_count} | Duración: {recording_duration:.2f}s")
        
        # Liberar el VideoWriter
        if current_video_writer is not None:
            current_video_writer.release()
            current_video_writer = None
        
        # Limpiar señal para próxima iteración
        start_recording.clear()
    
    print("[RECORDER] Worker finalizado.")


def prepare_video_writer(output_path, width, height):
    """Prepara un nuevo VideoWriter para la siguiente iteración"""
    global current_video_writer, current_output_path
    
    print(f"[SETUP] Preparando VideoWriter: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    if not writer.isOpened():
        print("[SETUP] ⚠️  Falló 'mp4v', intentando 'XVID' (.avi)")
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    if not writer.isOpened():
        print("[SETUP] ❌ Error fatal: No se pudo inicializar VideoWriter")
        return False
    
    current_video_writer = writer
    current_output_path = output_path
    print("[SETUP] ✓ VideoWriter listo")
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


def run_experiment_iteration(cap_experiment, exp_fps, nombre_persona, numero_intento, 
                             exp_width, exp_height, save_path, is_first_iteration):
    """Ejecuta UNA iteración del experimento"""
    global recording_active
    
    print("\n" + "="*70)
    print(f"   INTENTO {numero_intento}")
    print("="*70)
    
    # Preparar VideoWriter para esta iteración
    output_filename = f"{nombre_persona}_intento_{numero_intento}.mp4"
    output_video_path = os.path.join(save_path, output_filename)
    
    if not prepare_video_writer(output_video_path, cam_width, cam_height):
        print(f"[INTENTO {numero_intento}] ❌ Error al preparar grabación. Saltando iteración.")
        return False
    
    # Mensaje de preparación (solo después de la primera iteración)
    if not is_first_iteration:
        show_countdown("Experiment Video", "Prepárate para el siguiente", 2000)
    
    # Cuenta regresiva
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
    
    # Purgar cola antes de empezar
    purged = purge_frame_queue()
    print(f"[INTENTO {numero_intento}] Cola purgada ({purged} frames)")
    
    # Reiniciar video al inicio
    cap_experiment.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"\n{'⚡'*35}")
    print(f"   ¡INICIANDO INTENTO {numero_intento}!")
    print(f"{'⚡'*35}\n")
    
    # Reproducir experimento y grabar
    next_frame_time = time.time()
    first_frame_shown = False
    
    while True:
        current_time = time.time()
        
        # Control de timing
        if current_time < next_frame_time:
            delay_ms = int((next_frame_time - current_time) * 1000)
            if delay_ms > 0:
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q'):
                    print(f"[INTENTO {numero_intento}] Detenido manualmente.")
                    recording_active = False
                    return False
                continue
        
        next_frame_time = current_time + (1.0 / exp_fps)
        
        # Leer frame del experimento
        ret_exp, frame_exp = cap_experiment.read()
        if not ret_exp:
            print(f"[INTENTO {numero_intento}] ✓ Video finalizado.")
            break
        
        # ¡SINCRONIZACIÓN! Activar grabación en el primer frame
        if not first_frame_shown:
            recording_active = True
            start_recording.set()  # Señal al worker
            first_frame_shown = True
        
        # Mostrar frame del experimento
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
            print(f"[INTENTO {numero_intento}] Detenido manualmente.")
            recording_active = False
            return False
    
    # Detener grabación
    print(f"\n{'🛑'*35}")
    print(f"   FINALIZANDO INTENTO {numero_intento}")
    print(f"{'🛑'*35}\n")
    
    recording_active = False
    
    # Esperar a que el worker termine de grabar
    time.sleep(0.5)
    
    print(f"[INTENTO {numero_intento}] ✓ Completado\n")
    return True


def run_all_experiments(nombre_persona, total_iteraciones):
    """Función principal que ejecuta TODAS las iteraciones en un solo flujo"""
    global capture_active, recording_active
    
    print("\n" + "="*70)
    print("   INICIANDO SESIÓN DE EXPERIMENTOS")
    print("="*70)
    
    # Configurar rutas
    save_path = f"/home/vit/Documentos/Tesis3D/Videos/Experimento_{EXP_NUM}/{nombre_persona}"
    experiment_video_path = f"/home/vit/Documentos/Tesis3D/Videos/Animaciones_experimentos/experimento_{EXP_NUM}.mp4"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Cargar video del experimento
    cap_experiment = cv2.VideoCapture(experiment_video_path)
    if not cap_experiment.isOpened():
        print(f"❌ Error: No se pudo abrir {experiment_video_path}")
        return
    
    exp_width = int(cap_experiment.get(cv2.CAP_PROP_FRAME_WIDTH))
    exp_height = int(cap_experiment.get(cv2.CAP_PROP_FRAME_HEIGHT))
    exp_fps = cap_experiment.get(cv2.CAP_PROP_FPS)
    if exp_fps <= 0:
        exp_fps = 30.0
    
    print(f"[VIDEO] Experimento: {exp_width}x{exp_height} @ {exp_fps} FPS")
    
    # ==================================================================
    # INICIALIZACIÓN ÚNICA - Se hace UNA SOLA VEZ
    # ==================================================================
    
    print("\n" + "-"*70)
    print("   INICIALIZANDO SISTEMA")
    print("-"*70)
    
    # 1. Iniciar captura continua
    capture_thread = Thread(target=capture_webcam_stream, daemon=True)
    capture_thread.start()
    time.sleep(1.5)
    
    if not capture_active:
        print("❌ Error: La webcam no pudo iniciarse.")
        cap_experiment.release()
        return
    
    # 2. Iniciar worker de grabación
    recording_thread = Thread(target=recording_worker, daemon=True)
    recording_thread.start()
    
    print(f"[SISTEMA] ✓ Webcam activa: {cam_width}x{cam_height}")
    print("[SISTEMA] ✓ Worker de grabación activo")
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
        success = run_experiment_iteration(
            cap_experiment, exp_fps, nombre_persona, str(i),
            exp_width, exp_height, save_path, 
            is_first_iteration=(i == 1)
        )
        
        if not success:
            print(f"\n⚠️  Sesión interrumpida en el intento {i}")
            break
        
        # Pausa entre iteraciones (excepto en la última)
        if i < total_iteraciones:
            time.sleep(1)
    
    # ==================================================================
    # LIMPIEZA FINAL
    # ==================================================================
    
    print("\n" + "="*70)
    print("   FINALIZANDO SESIÓN")
    print("="*70)
    
    recording_active = False
    capture_active = False
    start_recording.set()  # Liberar worker si está esperando
    
    print("[CLEANUP] Esperando finalización de hilos...")
    capture_thread.join(timeout=2.0)
    recording_thread.join(timeout=3.0)
    
    cap_experiment.release()
    cv2.destroyAllWindows()
    
    print("\n" + "✅"*35)
    print("   TODAS LAS ITERACIONES COMPLETADAS")
    print("✅"*35 + "\n")


# ==================================================================
# PUNTO DE ENTRADA
# ==================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   SISTEMA DE EXPERIMENTOS - FLUJO CONTINUO")
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
    print(f"✓ Iteraciones: {total_iteraciones}")
    print("-"*70)
    
    input("\nPresiona ENTER para comenzar...")
    
    run_all_experiments(nombre_persona, total_iteraciones)
    
    print("\n🎉 Sesión finalizada. ¡Gracias!\n")