import cv2
import os
import time
import numpy as np
import queue
from threading import Thread

# Configuración del ESP32
ESP32_URL = "http://192.168.0.19"
ESP32_STREAM_URL = ESP32_URL + ":81/stream"

# Cola para almacenar frames de la ESP32-CAM
frame_queue = queue.Queue(maxsize=60)  # Limitar el tamaño de la cola
screen_width, screen_height = 1920, 1080

# --- MODIFICACIÓN ---
# Estas variables se establecerán dinámicamente, pero se mantienen como globales
# para que 'capture_esp32_stream' pueda actualizarlas.
cam_height, cam_width = 240, 320 

# Variables de control
recording_active = False
capture_active = True

# Función para capturar frames del ESP32 y ponerlos en la cola
def capture_esp32_stream():
    # --- MODIFICACIÓN ---
    # Se añade 'global' para asegurar que actualice las variables globales
    global cam_width, cam_height, capture_active, frame_queue
    
    print(f"Conectando al stream del ESP32 en: {ESP32_STREAM_URL}")
    
    cap_esp32 = cv2.VideoCapture(ESP32_STREAM_URL)
    
    if not cap_esp32.isOpened():
        print("Error: No se pudo conectar al stream del ESP32.")
        return
    
    # Obtener las dimensiones del stream
    ret, first_frame = cap_esp32.read()
    if ret:
        cam_height, cam_width, _ = first_frame.shape
        print(f"Resolución del ESP32: {cam_width}x{cam_height}")
        # Poner el primer frame en la cola
        if not frame_queue.full():
            frame_queue.put(first_frame)
    else:
        print("Advertencia: No se pudo leer el primer frame del ESP32.")
    
    frame_count = 0
    last_fps_print_time = time.time()
    
    while capture_active:
        ret, frame = cap_esp32.read()
        
        if not ret:
            print("Error al leer frame del ESP32. Intentando reconectar...")
            cap_esp32.release()
            time.sleep(1)
            cap_esp32 = cv2.VideoCapture(ESP32_STREAM_URL)
            continue
        
        # Si la cola está llena, descartar el frame más antiguo
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        frame_queue.put(frame)
        frame_count += 1
        
        # Imprimir FPS aproximados cada 5 segundos
        current_time = time.time()
        if current_time - last_fps_print_time >= 5:
            fps = frame_count / (current_time - last_fps_print_time)
            print(f"ESP32-CAM captura FPS: {fps:.2f}")
            frame_count = 0
            last_fps_print_time = current_time
    
    cap_esp32.release()
    print("Stream del ESP32 detenido.")

# --- MODIFICACIÓN ---
# La función ahora acepta el path de salida como argumento
def record_esp32_frames(output_video_path):
    global recording_active, capture_active, frame_queue
    
    try:
        # Esperar a que recording_active se active
        print("Thread de grabación esperando señal para iniciar...")
        while not recording_active and capture_active:
            time.sleep(0.1)
        
        if not capture_active:
            print("Grabación cancelada antes de iniciar.")
            return
        
        print("¡INICIANDO GRABACIÓN DE ESP32-CAM!")
        
        # Obtener el primer frame para la grabación
        first_frame = None
        timeout_counter = 0
        
        while first_frame is None and timeout_counter < 10:
            try:
                first_frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                print("Esperando frames de la ESP32-CAM...")
                timeout_counter += 1
                if not recording_active:
                    print("Grabación cancelada.")
                    return
        
        if first_frame is None:
            print("Error: No se recibió ningún frame de la ESP32-CAM.")
            return
        
        # --- MODIFICACIÓN ---
        # Obtener dimensiones directamente del primer frame (más robusto)
        rec_height, rec_width, _ = first_frame.shape
        print(f"Resolución de grabación (detectada): {rec_width}x{rec_height}")
        
        # Inicializar el escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Usar el path de salida del argumento
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (rec_width, rec_height))
        
        # --- MODIFICACIÓN ---
        # Lógica de fallback para el path de salida
        if not out.isOpened():
            print("Advertencia: Falló 'mp4v'. Intentando con 'XVID' (.avi)")
            output_video_path = output_video_path.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (rec_width, rec_height))
            if not out.isOpened():
                print("Advertencia: Falló 'XVID'. Intentando con 'MJPG' (.avi)")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (rec_width, rec_height))
        
        if not out.isOpened():
            print(f"Error fatal: No se pudo abrir el VideoWriter con ningún códec para: {output_video_path}")
            return

        print(f"Guardando video en: {output_video_path} con resolución {rec_width}x{rec_height}")
        
        # Escribir el primer frame
        out.write(first_frame)
        
        # Contador de frames escritos
        frame_count = 1
        last_fps_print_time = time.time()
        
        # Grabar frames de la ESP32-CAM mientras recording_active sea True
        while recording_active:
            try:
                # Intentar obtener un frame con timeout
                frame = frame_queue.get(timeout=0.5)
                out.write(frame)
                frame_count += 1
                
                # Imprimir FPS aproximados cada 5 segundos
                current_time = time.time()
                if current_time - last_fps_print_time >= 5:
                    fps = frame_count / (current_time - last_fps_print_time)
                    print(f"ESP32-CAM grabación FPS: {fps:.2f} - Frames grabados: {frame_count}")
                    last_fps_print_time = current_time
                
            except queue.Empty:
                # No hay frames disponibles en este momento, continuar esperando
                continue
        
        # Limpiar y cerrar el video
        out.release()
        print(f"Video de ESP32-CAM guardado exitosamente en: {output_video_path}")
        print(f"Total de frames grabados: {frame_count}")
    
    except Exception as e:
        print(f"Error en el hilo de grabación: {e}")

# --- MODIFICACIÓN ---
# La función ahora acepta el nombre y el número de intento
def run_experiment(nombre_persona, numero_intento):
    global recording_active, capture_active, frame_queue
    
    # --- INICIO DE MODIFICACIÓN: Resetear estado ---
    print("\n" + "*"*60)
    print(f"INICIANDO EXPERIMENTO: {nombre_persona} - Intento {numero_intento}")
    print("*"*60)

    # Resetear variables de control para esta iteración
    recording_active = False
    capture_active = True
    
    # Limpiar la cola de frames de la ejecución anterior
    print("Limpiando cola de frames...")
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    print("Cola de frames limpiada.")
    # --- FIN DE MODIFICACIÓN ---

    # --- MODIFICACIÓN ---
    # Se eliminaron las solicitudes de input, ya que vienen por parámetros
    
    # Rutas de los archivos
    Save_video_path = f"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/{nombre_persona}"

    #Ruta experimento 1
    experiment_video_path = "/home/vit/Documentos/Tesis3D/Videos/Animaciones_experimentos/experimento_1.mp4"
    
    #Ruta experimento 2
    #experiment_video_path = "/home/vit/Documentos/Tesis3D/Videos/Animaciones_experimentos/experimento_2.mp4"

    #Ruta experimento 3
    #experiment_video_path = "/home/vit/Documentos/Tesis3D/Videos/Animaciones_experimentos/experimento_3.mp4"

    # Crear el directorio de destino si no existe
    if not os.path.exists(Save_video_path):
        os.makedirs(Save_video_path)

    # Captura de video del experimento
    cap_experiment = cv2.VideoCapture(experiment_video_path)
    
    # Verificar que la captura se haya iniciado correctamente
    if not cap_experiment.isOpened():
        print(f"Error: No se pudo abrir el video del experimento en: {experiment_video_path}")
        capture_active = False
        return # Usar return en lugar de exit() para no detener el bucle
    
    # Obtener las dimensiones del video de experimento
    exp_width = int(cap_experiment.get(cv2.CAP_PROP_FRAME_WIDTH))
    exp_height = int(cap_experiment.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Obtener los FPS del video del experimento para reproducirlo a la velocidad correcta
    exp_fps = cap_experiment.get(cv2.CAP_PROP_FPS)
    if exp_fps <= 0:
        exp_fps = 30.0  # Valor predeterminado si no se puede determinar
    
    print(f"Video del experimento: {exp_width}x{exp_height} @ {exp_fps} FPS")
    
    # Iniciar el hilo de captura del ESP32
    capture_thread = Thread(target=capture_esp32_stream, daemon=True)
    capture_thread.start()
    
    # --- INICIO DE MODIFICACIÓN: Pasar path al thread de grabación ---
    # Construir el path de salida EXACTAMENTE como lo pediste
    output_filename = f"{nombre_persona}_intento_{numero_intento}.mp4"
    output_video_path = os.path.join(Save_video_path, output_filename)
    
    # Iniciar el hilo de grabación (estará en espera)
    recording_thread = Thread(target=record_esp32_frames, args=(output_video_path,), daemon=True)
    # --- FIN DE MODIFICACIÓN ---
    recording_thread.start()
    
    # Dar tiempo para que se establezca la conexión
    print("Esperando conexión con ESP32-CAM...")
    time.sleep(2)
    
    # Configurar la ventana para mostrar el video del experimento
    cv2.namedWindow("Experiment Video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Experiment Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Crear la cuenta regresiva antes de iniciar la reproducción
    print("Iniciando cuenta regresiva...")
    
    # Crea un fondo negro del tamaño de la pantalla
    countdown_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    for count in range(3, 0, -1):
        # Limpia el fondo
        countdown_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        # Configura el texto
        text = str(count)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 10
        font_thickness = 25
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # Calcula la posición central
        text_x = (screen_width - text_size[0]) // 2
        text_y = (screen_height + text_size[1]) // 2
        
        # Dibuja el número en el centro de la pantalla
        cv2.putText(countdown_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # Muestra el frame
        cv2.imshow("Experiment Video", countdown_frame)
        cv2.waitKey(1000)  # Espera 1 segundo
    
    # Muestra "¡Comenzando!" después de la cuenta regresiva
    countdown_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    text = "Comenzando.."
    font_scale = 5
    font_thickness = 10
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (screen_width - text_size[0]) // 2
    text_y = (screen_height + text_size[1]) // 2
    cv2.putText(countdown_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.imshow("Experiment Video", countdown_frame)
    cv2.waitKey(1000)  # Espera 1 segundo
    
    print(f"Reproduciendo experimento (Intento {numero_intento})...")
    print("Presiona 'q' para detener ESTA iteración.")
    
    # ¡ACTIVAR LA GRABACIÓN AHORA! - Justo antes de mostrar el primer frame del video
    print("\n" + "="*50)
    print(f"ACTIVANDO GRABACIÓN (Intento {numero_intento})")
    print("="*50 + "\n")
    recording_active = True
    
    # Control de tiempo para reproducir el video a velocidad real
    next_frame_time = time.time()
    
    # Bucle principal de reproducción del video del experimento
    while cap_experiment.isOpened():
        # Controlar el tiempo para mantener la velocidad correcta de reproducción
        current_time = time.time()
        if current_time < next_frame_time:
            # Esperar hasta que sea el momento de mostrar el siguiente frame
            delay_ms = int((next_frame_time - current_time) * 1000)
            if delay_ms > 0:
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q'):
                    print("Experimento detenido manualmente.")
                    break
                continue
        
        # Calcular el tiempo para el próximo frame
        next_frame_time = current_time + (1.0 / exp_fps)
        
        # Leer el siguiente frame del video del experimento
        ret_exp, frame_exp = cap_experiment.read()
        if not ret_exp:
            print(f"Video del experimento (Intento {numero_intento}) finalizado.")
            break
        
        # Mostrar el video del experimento a pantalla completa
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
        
        # Usar waitKey(1) solo para procesar eventos, no para timing
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Experimento (Intento {numero_intento}) detenido manualmente.")
            break
    
    # Detener la grabación cuando termina el experimento
    print("\n" + "="*50)
    print(f"DETENIENDO GRABACIÓN (Intento {numero_intento})")
    print("="*50 + "\n")
    recording_active = False
    capture_active = False
    
    print("Esperando a que termine la grabación...")
    # Esperar a que los hilos terminen
    capture_thread.join(timeout=2.0)
    recording_thread.join(timeout=5.0)
    
    cap_experiment.release()
    cv2.destroyAllWindows()
    
    print(f"Intento {numero_intento} finalizado.")
    print("Esperando 3 segundos antes de la siguiente iteración...")
    time.sleep(3) # Pausa entre experimentos

if __name__ == "__main__":
    # --- INICIO DE MODIFICACIÓN: Lógica de bucle principal ---
    
    # Solicitar datos por consola UNA VEZ
    print("=== CONFIGURACIÓN DE LA SERIE DE EXPERIMENTOS ===")
    nombre_persona = input("Ingresa el nombre de la persona: ").strip()
    while not nombre_persona:
        print("El nombre no puede estar vacío.")
        nombre_persona = input("Ingresa el nombre de la persona: ").strip()

    total_iteraciones_str = input("Ingresa el NÚMERO TOTAL de iteraciones (ej: 3): ").strip()
    while not total_iteraciones_str.isdigit() or int(total_iteraciones_str) <= 0:
        print("El número de iteraciones debe ser un número positivo.")
        total_iteraciones_str = input("Ingresa el NÚMERO TOTAL de iteraciones (ej: 3): ").strip()
    
    total_iteraciones = int(total_iteraciones_str)

    print("\nConfiguración:")
    print(f"Persona: {nombre_persona}")
    print(f"Total de intentos: {total_iteraciones}")
    print("-" * 40)
    
    # Bucle principal para ejecutar el experimento N veces
    for i in range(1, total_iteraciones + 1):
        numero_intento_actual = str(i)
        
        # Ejecutar el experimento completo
        run_experiment(nombre_persona, numero_intento_actual)
    
    print("\n" + "="*60)
    print("TODAS LAS ITERACIONES HAN FINALIZADO.")
    print("="*60)
    # --- FIN DE MODIFICACIÓN ---