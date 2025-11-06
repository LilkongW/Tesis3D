import cv2
import os

# --- Configuración ---
nombre = "Geremy"
# 1. Carpeta que contiene tus videos fuente
input_folder = rf"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/{nombre}"

# 2. Carpeta donde se guardarán TODOS los frames
output_folder = r'/home/vit/Documentos/Tesis3D/Videos'

# 3. Intervalo de frames (guardar 1 cada 5)
frame_interval = 120

# 4. Extensiones de video que queremos procesar (en minúsculas)
video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.mpg')

# ---------------------

def procesar_carpeta_videos(input_dir, output_dir, interval, valid_extensions):
    """
    Recorre una carpeta de entrada, extrae frames de todos los videos
    que encuentra y los guarda en una carpeta de salida.
    """
    
    # --- 1. Preparar la carpeta de salida ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Carpeta de salida creada: {output_dir}")
    else:
        print(f"Usando carpeta de salida existente: {output_dir}")

    # --- 2. Recorrer la carpeta de entrada ---
    print(f"\nBuscando videos en: {input_dir}...")
    
    try:
        # Listar todos los archivos en el directorio de entrada
        archivos_en_carpeta = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"Error: La carpeta de entrada '{input_dir}' no existe.")
        print("Por favor, crea la carpeta o corrige la ruta en 'input_folder'.")
        return
    
    videos_procesados = 0

    # Iterar sobre cada archivo encontrado
    for filename in archivos_en_carpeta:
        
        # Comprobar si el archivo tiene una extensión de video válida
        if not filename.lower().endswith(valid_extensions):
            # print(f"Ignorando archivo (no es video): {filename}")
            continue
            
        # --- 3. Procesar el archivo de video ---
        videos_procesados += 1
        video_path = os.path.join(input_dir, filename)
        
        # Obtener el nombre base del video (sin la extensión)
        # ej: 'trial_01.mp4' -> 'trial_01'
        video_name_base = os.path.splitext(filename)[0]
        
        print(f"\n--- Procesando video ({videos_procesados}): {filename} ---")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video '{video_path}'. Saltando...")
            continue

        frame_count = 0  # Contador total de frames leídos (para este video)
        saved_count = 0  # Contador de frames guardados (para este video)

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Se terminó este video

            # Lógica de extracción (1 cada 'interval' frames)
            if frame_count % interval == 0:
                
                # Nombre de archivo de salida único
                # ej: "trial_01_frame_000005.jpg"
                output_filename = f"{video_name_base}_frame_{frame_count:06d}.jpg"
                save_path = os.path.join(output_dir, output_filename)
                
                cv2.imwrite(save_path, frame)
                saved_count += 1
            
            frame_count += 1

        cap.release()
        print(f"Video '{filename}' completado.")
        print(f"Frames leídos: {frame_count} | Frames guardados: {saved_count}")

    # --- 4. Finalizar ---
    print("\n--- Proceso General Completado ---")
    if videos_procesados == 0:
        print(f"No se encontraron videos con extensiones {valid_extensions} en '{input_dir}'.")
    else:
        print(f"Total de videos procesados: {videos_procesados}")

# --- Ejecutar la función ---
if __name__ == "__main__":
    procesar_carpeta_videos(input_folder, output_folder, frame_interval, video_extensions)