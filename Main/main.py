import os
import csv  # <-- Importar CSV
# Importamos la única función que necesitamos desde nuestro módulo de utilidades
from utils.eye_tracker_utils import process_video_from_path

def main():
    # --- CONFIGURACIÓN DE RUTAS ---
    
    # Carpeta que contiene los videos a procesar
    VIDEO_FOLDER_PATH = r"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/ManuelMal/ROI_videos_640x480"  # <-- RUTA DE VIDEOS
    
    # Carpeta donde se guardarán los archivos CSV generados
    CSV_OUTPUT_PATH = r"/home/vit/Documentos/Tesis3D/Data/ManuelMal_data"  # <-- RUTA DE SALIDA DE CSV

    # ---------------------------------
    
    # Extensiones de video comunes a buscar
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    if not os.path.isdir(VIDEO_FOLDER_PATH):
        print(f"Error: La carpeta de videos no existe en: {VIDEO_FOLDER_PATH}")
        return
        
    # Crear la carpeta de salida de CSV si no existe
    try:
        os.makedirs(CSV_OUTPUT_PATH, exist_ok=True)
        print(f"Guardando archivos CSV en: {CSV_OUTPUT_PATH}")
    except OSError as e:
        print(f"Error al crear la carpeta de salida {CSV_OUTPUT_PATH}: {e}")
        return
    
    print(f"Buscando videos en: {VIDEO_FOLDER_PATH}")
    
    # Recorrer todos los archivos en la carpeta
    video_files_found = []
    for filename in os.listdir(VIDEO_FOLDER_PATH):
        # Obtener la extensión del archivo y convertirla a minúsculas
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Comprobar si la extensión está en nuestra lista de videos
        if file_ext in video_extensions:
            video_files_found.append(filename)
            
    if not video_files_found:
        print(f"No se encontraron archivos de video en la carpeta: {VIDEO_FOLDER_PATH}")
        return

    print(f"Se encontraron {len(video_files_found)} videos. Comenzando procesamiento...")

    # Procesar cada video encontrado
    for video_name in video_files_found:
        # Construir la ruta completa al archivo de video
        full_video_path = os.path.join(VIDEO_FOLDER_PATH, video_name)
        
        # --- Creación de la ruta del CSV ---
        # Tomar el nombre base del video (ej. "video_1_ROI")
        video_name_base = os.path.splitext(video_name)[0]
        # Crear un nombre de archivo CSV (ej. "video_1_ROI_data.csv")
        csv_file_name = f"{video_name_base}_data.csv"
        # Crear la ruta completa de salida para el CSV
        full_csv_path = os.path.join(CSV_OUTPUT_PATH, csv_file_name)
        
        print(f"\n--- [INICIANDO] Procesando: {video_name} ---")
        
        # Llamar a la función de procesamiento, pasando las nuevas rutas
        process_video_from_path(full_video_path, video_name, full_csv_path)
        
        print(f"--- [COMPLETADO] Video: {video_name} ---")
    
    print("\nTodos los videos en la carpeta han sido procesados.")

if __name__ == "__main__":
    main()