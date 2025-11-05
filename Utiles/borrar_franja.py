import cv2
import os
import numpy as np
import time

# --- Configuración ---

# 1. Carpeta que contiene los videos a MODIFICAR
# ¡¡ESTOS VIDEOS SERÁN SOBRESCRITOS!!
input_folder = '/home/vit/Documentos/Tesis3D/Videos/Experimento_3/Majo'

# 2. Altura de la franja a reemplazar (en píxeles desde la parte superior)
crop_height = 8

# 3. Color de la franja a reemplazar: Blanco (BGR)
fill_color = [255, 255, 255] 

# 4. Extensiones de video que queremos procesar (en minúsculas)
video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.mpg')

# ---------------------

def sobrescribir_videos_franja(video_dir, pixels_to_replace, color, valid_extensions):
    """
    Recorre una carpeta de videos, reemplaza una franja superior
    con una barra blanca y SOBRESCRIBE los archivos originales.
    
    ADVERTENCIA: Esta operación es destructiva.
    """
    
    print(f"ADVERTENCIA: Se modificarán y sobrescribirán los archivos en: {video_dir}")
    print("Tienes 5 segundos para cancelar (Ctrl+C)...")
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nProceso cancelado por el usuario.")
        return
        
    print("\nIniciando proceso...")

    # --- 1. Recorrer la carpeta de entrada ---
    try:
        archivos_en_carpeta = os.listdir(video_dir)
    except FileNotFoundError:
        print(f"Error: La carpeta de entrada '{video_dir}' no existe.")
        return
    
    videos_procesados = 0
    videos_fallidos = 0

    for filename in archivos_en_carpeta:
        
        if not filename.lower().endswith(valid_extensions):
            continue
            
        videos_procesados += 1
        video_path = os.path.join(video_dir, filename)
        
        # --- 2. Definir ruta temporal ---
        # Usamos os.path.splitext para separar nombre y extensión
        name_base, ext = os.path.splitext(filename)
        temp_filename = f"{name_base}__temp__{ext}"
        temp_output_path = os.path.join(video_dir, temp_filename)
        
        print(f"\n--- Procesando (sobrescribiendo): {filename} ---")
        
        try:
            # --- 3. Abrir el video y obtener propiedades ---
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: No se pudo abrir '{filename}'. Saltando...")
                videos_fallidos += 1
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if height <= pixels_to_replace:
                print("Error: Video demasiado bajo. Saltando...")
                cap.release()
                videos_fallidos += 1
                continue

            # --- 4. Definir el escritor (al archivo temporal) ---
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            # --- 5. Procesar frame por frame ---
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Fin del video

                # Reemplazar la franja por color blanco
                frame[0:pixels_to_replace, :] = color
                out.write(frame)

            # --- 6. Cerrar ambos archivos (MUY IMPORTANTE) ---
            cap.release()
            out.release()
            
            # --- 7. Reemplazar el archivo original ---
            os.remove(video_path)               # Borrar el original
            os.rename(temp_output_path, video_path) # Renombrar el temporal
            
            print(f"Video '{filename}' sobrescrito exitosamente.")

        except Exception as e:
            print(f"Error inesperado procesando '{filename}': {e}")
            videos_fallidos += 1
            # Limpieza en caso de error
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals() and out.isOpened():
                out.release()
            # Borrar el archivo temporal si se creó pero falló
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

    # --- 8. Finalizar ---
    print("\n--- Proceso General Completado ---")
    total = videos_procesados + videos_fallidos
    if total == 0:
        print(f"No se encontraron videos con extensiones {valid_extensions} en '{video_dir}'.")
    else:
        print(f"Videos procesados (sobrescritos): {videos_procesados}")
        print(f"Videos omitidos (con error): {videos_fallidos}")

# --- Ejecutar la función ---
if __name__ == "__main__":
    # Asegúrate de que esta es la carpeta correcta
    sobrescribir_videos_franja(input_folder, crop_height, fill_color, video_extensions)