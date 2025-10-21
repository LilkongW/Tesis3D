import cv2
import requests
import datetime
import os # Importar para manejo de rutas de archivo

# ESP32 URL
URL = "http://192.168.0.19"
AWB = True

# --- Variables Globales para Grabación ---
recording = False
video_writer = None
OUTPUT_FOLDER = "grabaciones"  # Carpeta donde se guardarán los videos
# El FPS debe ser consistente. Si el stream del ESP32 no es fijo, 15 FPS es un valor seguro.
FPS = 30 


# Face recognition and opencv setup
# Asegúrate de que el URL esté correcto y el stream esté activo
cap = cv2.VideoCapture(URL + ":81/stream")


def set_resolution(url: str, index: int = 1, verbose: bool = False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("Resoluciones disponibles:\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Índice incorrecto.")
    except:  # noqa: E722
        print("SET_RESOLUTION: Algo salió mal.")


def set_quality(url: str, value: int = 1, verbose: bool = False):
    try:
        if value >= 10 and value <= 63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:  # noqa: E722
        print("SET_QUALITY: Algo salió mal.")


def set_contrast(url: str, value: int = 0):
    try:
        # El contraste normalmente va de -2 a 2, pero en algunos ESP32 puede ser de 0 a 4
        # Ajustamos para que 0 sea el valor por defecto
        if value >= -2 and value <= 2:
            requests.get(url + "/control?var=contrast&val={}".format(value))
            print(f"Contraste establecido a: {value}")
        else:
            print("Valor de contraste no válido. Debe estar entre -2 y 2")
    except:  # noqa: E722
        print("SET_CONTRAST: Algo salió mal.")


def set_awb(url: str, awb: int = 1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:  # noqa: E722
        print("SET_AWB: Algo salió mal.")
    return awb


# --- Función para iniciar la grabación ---
def start_recording(frame_size):
    global recording, video_writer
    
    # 1. Crear la carpeta de salida si no existe
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    # 2. Generar nombre de archivo basado en la hora
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_FOLDER, f"video_{timestamp}.mp4")

    # 3. Definir el codec (ej. 'mp4v' para .mp4 o 'XVID')
    # Nota: Es crucial tener codecs compatibles con tu instalación de OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # 4. Inicializar VideoWriter
    video_writer = cv2.VideoWriter(filename, fourcc, FPS, frame_size)
    recording = True
    print(f"\n--- INICIANDO GRABACIÓN: {filename} ---")

# --- Función para detener la grabación ---
def stop_recording():
    global recording, video_writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        recording = False
        print("--- GRABACIÓN DETENIDA ---")


if __name__ == "__main__":
    # CONFIGURACIÓN INICIAL: Establecer resolución VGA y contraste 0
    # Obtener el tamaño del frame después de inicializar la cámara
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width, _ = frame.shape
            frame_size = (width, height)
            print(f"Tamaño del frame: {width}x{height}")
        else:
            # Asumir una resolución VGA si la lectura falla
            frame_size = (640, 480) 
            print("Advertencia: No se pudo leer el primer frame. Asumiendo tamaño VGA (640x480) para la grabación.")
    else:
        print("Error: No se pudo conectar al stream. Saliendo.")
        exit()

    while True:
        if cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Error al leer el frame.")
                break
                

                        
            cv2.imshow("Stream del ESP32", frame)
            
            # --- Escribir el frame si se está grabando ---
            if recording and video_writer is not None:
                video_writer.write(frame)

            key = cv2.waitKey(1)

            if key == ord("r"):
                idx = int(input("Select resolution index: "))
                set_resolution(URL, index=idx, verbose=True)

            elif key == ord("c"):
                val = int(input("Set contrast (-2 to 2): "))
                set_contrast(URL, value=val)

            elif key == ord("a"):
                AWB = set_awb(URL, AWB)

            # --- NUEVA OPCIÓN: INICIAR/DETENER GRABACIÓN ---
            elif key == ord("g"):
                if not recording:
                    start_recording(frame_size)
                else:
                    stop_recording()

            elif key == 27:  # Tecla ESC
                break
        else:
            print("Stream no abierto. Intentando reconectar...")
            cap = cv2.VideoCapture(URL + ":81/stream")
            cv2.waitKey(5000) # Espera 5 segundos antes de intentar de nuevo
            
    # --- Limpieza al salir ---
    if recording:
        stop_recording()
    
    cv2.destroyAllWindows()
    cap.release()