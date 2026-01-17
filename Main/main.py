import os
from utils.eye_tracker_utils import process_video_from_path, set_config

def main():
    # ==========================================
    #    CONFIGURACIÓN DE PARÁMETROS (USUARIO)
    # ==========================================
    
    # Variables de control del procesamiento
    ENABLE_IRIS_PROCESSING = False   # True: Calcula iris. False: Solo pupila (más rápido)
    SHOW_VISUALIZATION = False       # True: Muestra ventana. False: Modo rápido
    YOLO_MIN_CONFIDENCE = 0.7        # Confianza mínima YOLO (0.0 - 1.0)
    PUPIL_FIXED_THRESHOLD = 14       # Umbral de binarización para pupila
    MAX_INTERSECTION_DISTANCE = 10   # Distancia máxima para intersecciones
    
    # Aplicar configuración al módulo eye_tracker_utils
    set_config(
        enable_iris=ENABLE_IRIS_PROCESSING,
        show_viz=SHOW_VISUALIZATION,
        yolo_conf=YOLO_MIN_CONFIDENCE,
        pupil_thresh=PUPIL_FIXED_THRESHOLD,
        max_intersect=MAX_INTERSECTION_DISTANCE
    )
    
    # ==========================================
    #       CONFIGURACIÓN DE EXPERIMENTO
    # ==========================================
    
    prev = (150, 170)  # Centro inicial del modelo (ancho, alto)
    
    EXP_NUM = 1        # Número de experimento
    NOMBRE = "test"  # Nombre del sujeto VictoriaRoso, Vielma, Stephanie, Sanchez
    
    # Ruta de videos - compatible con Windows y Linux
    VIDEO_FOLDER_PATH = os.path.join("Videos", f"Experimento_{EXP_NUM}", NOMBRE)
    
    # Carpeta donde se guardarán los archivos CSV generados
    CSV_OUTPUT_PATH = os.path.join("Data", f"Experimento_{EXP_NUM}", f"{NOMBRE}_data")
    
    # ==========================================
    #         PROCESAMIENTO DE VIDEOS
    # ==========================================
    
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
    
    print(f"\n{'='*60}")
    print("CONFIGURACIÓN ACTIVA:")
    print(f"{'='*60}")
    print(f"  • Procesamiento de Iris: {'ACTIVADO' if ENABLE_IRIS_PROCESSING else 'DESACTIVADO'}")
    print(f"  • Visualización: {'ACTIVADA' if SHOW_VISUALIZATION else 'DESACTIVADA'}")
    print(f"  • Confianza YOLO Mínima: {YOLO_MIN_CONFIDENCE}")
    print(f"  • Umbral de Pupila: {PUPIL_FIXED_THRESHOLD}")
    print(f"  • Distancia Máx. Intersección: {MAX_INTERSECTION_DISTANCE}")
    print(f"{'='*60}\n")
    
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
    
    print(f"Se encontraron {len(video_files_found)} videos. Comenzando procesamiento...\n")
    
    # Procesar cada video encontrado
    for idx, video_name in enumerate(video_files_found, 1):
        # Construir la ruta completa al archivo de video
        full_video_path = os.path.join(VIDEO_FOLDER_PATH, video_name)
        
        # --- Creación de la ruta del CSV ---
        video_name_base = os.path.splitext(video_name)[0]
        csv_file_name = f"{video_name_base}_data.csv"
        full_csv_path = os.path.join(CSV_OUTPUT_PATH, csv_file_name)
        
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(video_files_found)}] PROCESANDO: {video_name}")
        print(f"{'='*60}")
        
        # Llamar a la función de procesamiento
        process_video_from_path(full_video_path, video_name, full_csv_path, prev)
    
    print(f"\n{'='*60}")
    print("PROCESAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print("Todos los videos han sido procesados exitosamente.")
    print(f"Archivos CSV guardados en: {CSV_OUTPUT_PATH}\n")

if __name__ == "__main__":
    main()