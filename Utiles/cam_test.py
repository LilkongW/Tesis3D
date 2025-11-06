import cv2
import os
import time

# --- CONFIGURACIÓN DE GRABACIÓN ---
# Define dónde guardar los videos. 'os.path.expanduser("~")' es tu carpeta de usuario (ej. C:\Users\Victor o /home/vit)
OUTPUT_DIR = "/home/vit/Documentos/Tesis3D/Videos"
# Asegúrate de que la carpeta de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ------------------------------------

# Abre la cámara (0 = cámara por defecto, 1 = segunda cámara)
camara = cv2.VideoCapture(1)

if not camara.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("🎥 Presiona 'q' para salir.")
print("🔴 Presiona 'g' para Iniciar/Detener la grabación.")

# Variables para la grabación
is_recording = False
video_writer = None
# Usaremos 20.0 FPS. 'camara.get(cv2.CAP_PROP_FPS)' suele ser 0 para webcams.
fps_salida = 20.0 
# Códec de video (usa 'mp4v' para .mp4 o 'XVID' para .avi)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

while True:
    ret, frame = camara.read()
    if not ret:
        print("⚠️ No se pudo leer el frame.")
        break

    # --- Lógica de grabación ---
    
    # 1. Dibuja un indicador "REC" si estamos grabando
    if is_recording:
        
        # Escribe el frame en el archivo de video
        if video_writer is not None:
            video_writer.write(frame)

    # Muestra la imagen en una ventana
    cv2.imshow("Camara (Presiona 'g' para Grabar)", frame)

    # 2. Manejo de teclas
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
            
            # Obtiene el tamaño del frame (ancho, alto)
            h, w = frame.shape[:2]
            
            # Inicializa el VideoWriter
            video_writer = cv2.VideoWriter(video_path, fourcc, fps_salida, (w, h))
            print(f"🔴 ¡Iniciando grabación! Guardando en: {video_path}")


# --- Limpieza final ---

# Asegúrate de liberar el grabador si sales mientras grabas
if video_writer is not None:
    video_writer.release()
    print("Limpiando... (grabación finalizada)")

# Libera la cámara y cierra las ventanas
camara.release()
cv2.destroyAllWindows()