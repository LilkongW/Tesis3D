import cv2
import os
import time

# --- CONFIGURACIÓN DE GRABACIÓN ---
# Define dónde guardar los videos. 'os.path.expanduser("~")' es tu carpeta de usuario
OUTPUT_DIR = "/home/vit/Documentos/Tesis3D/Videos"
# Asegúrate de que la carpeta de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ------------------------------------

# --- CONFIGURACIÓN DE ALTA VELOCIDAD ---
# Fijamos la resolución a QVGA (320x240), el único modo con 120 FPS
TARGET_WIDTH = 320  # Ancho QVGA
TARGET_HEIGHT = 240 # Alto QVGA
TARGET_FPS = 120.0  # FPS deseado
# ------------------------------------

# Abre la cámara (1 = tu cámara USB)
# **CAMBIO CLAVE 1: Forzar el backend V4L2 de Linux**
camara = cv2.VideoCapture(1, cv2.CAP_V4L2)

if not camara.isOpened():
    print("❌ No se pudo acceder a la cámara. Revisa si el índice (1) es correcto.")
    exit()

# --- INTENTAR FIJAR RESOLUCIÓN Y FPS ---

# **CAMBIO CLAVE 2: Forzar el formato MJPG (Motion-JPEG)**
# V4L2-ctl confirmó que 120 FPS solo es posible con compresión MJPG.
try:
    fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
    camara.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
except Exception as e:
    print(f"⚠️ Error al intentar establecer el códec MJPG: {e}. Continuamos...")


# Solicitamos la configuración de baja resolución (QVGA) y alta velocidad.
camara.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
camara.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# Leer las configuraciones *reales* que el driver aceptó
actual_width = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps_driver = camara.get(cv2.CAP_PROP_FPS)

print(f"🎬 Solicitando: {TARGET_WIDTH}x{TARGET_HEIGHT} (QVGA) @ {TARGET_FPS} FPS")
print(f"✔️ Cámara configurada REALMENTE a: {actual_width}x{actual_height} @ {actual_fps_driver:.2f} FPS (según driver)")

print("\n🎥 Presiona 'q' para salir del programa.")
print("🔴 Presiona 'g' para Iniciar/Detener la grabación del video.")

# Variables para la grabación
is_recording = False
video_writer = None
# Usaremos 60.0 FPS para el archivo de salida (un valor común para alta velocidad)
fps_salida = 60.0 
# Códec de video para el archivo de salida (usa 'mp4v' para .mp4)
fourcc_salida = cv2.VideoWriter_fourcc(*'mp4v')

# --- VARIABLES PARA MEDIR EL RENDIMIENTO REAL (FPS) ---
frame_count = 0
time_start = time.time()
last_fps_update_time = time.time()
current_fps = 0.0
# ----------------------------------------------------

while True:
    # 1. Leer el frame
    # ¡cv2.waitKey(1) es crucial para mantener la alta velocidad!
    ret, frame = camara.read()
    if not ret:
        print("⚠️ No se pudo leer el frame. Deteniendo ciclo.")
        break
        
    # 2. Contador de frames para el cálculo del FPS promedio final
    frame_count += 1
    
    # 3. Cálculo de FPS instantáneo (actualiza cada 10 frames para mayor estabilidad)
    current_time = time.time()
    
    # Solo actualizamos el cálculo del FPS cada 10 frames para mayor estabilidad visual
    if frame_count % 10 == 0: 
        time_for_10_frames = current_time - last_fps_update_time
        if time_for_10_frames > 0:
            current_fps = 10 / time_for_10_frames
        # Reiniciar el contador de tiempo
        last_fps_update_time = current_time 

    
    # --- DIBUJAR ESTADO EN EL FRAME ---
    # Mostrar el FPS instantáneo
    cv2.putText(frame, f"FPS Actual: {current_fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Lógica de la grabación
    if is_recording:
        # Dibujar un círculo rojo y el texto REC
        cv2.circle(frame, (actual_width - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (actual_width - 70, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Escribe el frame en el archivo de video
        if video_writer is not None:
            video_writer.write(frame)

    # Muestra la imagen en una ventana
    cv2.imshow("Camara (Presiona 'g' para Grabar)", frame)

    # 4. Manejo de teclas: cv2.waitKey(1) es crucial para la alta velocidad
    key = cv2.waitKey(1) & 0xFF

    # Si presionas 'q', se cierra
    if key == ord('q'):
        break
        
    # Si presionas 'g', inicia o detiene la grabación
    elif key == ord('g'):
        if is_recording:
            # --- Detener grabación ---
            is_recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            print("✅ Grabación detenida.")
        else:
            # --- Iniciar grabación ---
            is_recording = True
            
            # Genera un nombre de archivo único con fecha y hora
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"grabacion_{timestamp}.mp4"
            video_path = os.path.join(OUTPUT_DIR, file_name)
            
            # Inicializa el VideoWriter (usamos la resolución y FPS de SALIDA)
            video_writer = cv2.VideoWriter(video_path, fourcc_salida, fps_salida, (actual_width, actual_height))
            print(f"🔴 ¡Iniciando grabación! Guardando en: {video_path}")


# --- Limpieza final y CÁLCULO DE FPS PROMEDIO (El resultado que buscabas) ---

# Calcula el tiempo total que estuvo encendida la cámara (desde el inicio hasta 'q')
time_end = time.time()
total_elapsed_time = time_end - time_start

# Asegúrate de liberar el grabador si sales mientras grabas
if video_writer is not None:
    video_writer.release()
    print("Limpiando... (grabación finalizada)")

# Si se procesó al menos un frame, calcula el promedio
if frame_count > 0 and total_elapsed_time > 0:
    average_fps = frame_count / total_elapsed_time
    print("-" * 50)
    print("📊 RESULTADOS FINALES DE RENDIMIENTO")
    print("-" * 50)
    print(f"   Frames totales procesados: {frame_count}")
    print(f"   Tiempo total transcurrido: {total_elapsed_time:.2f} segundos")
    print(f"   FPS PROMEDIO DURANTE EL SHOW: {average_fps:.2f} FPS")
    print("-" * 50)
else:
    print("No se procesaron frames o el tiempo fue cero.")


# Libera la cámara y cierra las ventanas
camara.release()
cv2.destroyAllWindows()