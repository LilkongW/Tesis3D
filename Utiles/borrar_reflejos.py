import cv2
import numpy as np
import os
import sys

# --- 📂 CONFIGURACIÓN ---
# ¡CUIDADO! Esta es la carpeta donde los videos serán MODIFICADOS PERMANENTEMENTE.
# Asegúrate de tener una copia de seguridad antes de ejecutar el script.
VIDEOS_FOLDER = r"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/prueba/ROI_videos_640x480"

# 2. Ajustes del Inpainting (un umbral más bajo es más agresivo).
REFLECTION_THRESHOLD = 170
# -------------------------

def remove_reflections(color_frame):
    """
    Detecta y elimina reflejos especulares brillantes de un fotograma a color mediante inpainting.

    Args:
        color_frame: El fotograma original a color (como un array de NumPy).

    Returns:
        El fotograma "inpainted" con los reflejos eliminados.
    """
    # Convertir el fotograma a escala de grises para encontrar fácilmente los puntos brillantes
    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    
    # Crear una máscara mediante umbralización para encontrar los píxeles muy brillantes
    _, mask = cv2.threshold(gray_frame, REFLECTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Dilatar la máscara para asegurar que el reflejo y su halo queden completamente cubiertos
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    # Aplicar inpainting en el FOTOGRAMA A COLOR ORIGINAL usando la máscara generada
    inpainted_frame = cv2.inpaint(color_frame, mask, 5, cv2.INPAINT_TELEA)
    
    return inpainted_frame

def process_and_overwrite_videos(target_dir):
    """
    Procesa todos los videos en un directorio para eliminar reflejos y los sobrescribe.
    """
    if not os.path.isdir(target_dir):
        print(f"❌ Error: El directorio no existe: '{target_dir}'")
        return

    # --- ADVERTENCIA DE SEGURIDAD ---
    print("====================================================================")
    print("🛑 ADVERTENCIA: ESTE SCRIPT SOBREESCRIBIRÁ LOS VIDEOS ORIGINALES. 🛑")
    print(f"   Directorio afectado: {target_dir}")
    print("   Esta acción no se puede deshacer.")
    print("   Asegúrate de tener una copia de seguridad antes de continuar.")
    print("====================================================================")
    
    try:
        # Espera a que el usuario confirme
        input("Presiona Enter para comenzar el proceso, o Ctrl+C para cancelar...")
    except KeyboardInterrupt:
        print("\n🚫 Proceso cancelado por el usuario.")
        return

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(target_dir) if os.path.splitext(f)[1].lower() in video_extensions]

    if not video_files:
        print(f"❌ No se encontraron archivos de video en '{target_dir}'")
        return

    print(f"\n✅ Se encontraron {len(video_files)} videos para procesar.")
    
    for video_filename in video_files:
        original_path = os.path.join(target_dir, video_filename)
        
        # Crear un nombre de archivo temporal en la misma carpeta
        name, ext = os.path.splitext(video_filename)
        temp_output_path = os.path.join(target_dir, f"{name}_temp{ext}")
        
        # Abrir el video original
        cap = cv2.VideoCapture(original_path)
        if not cap.isOpened():
            print(f"⚠️ No se pudo abrir el video: {video_filename}")
            continue
            
        # Obtener propiedades del video para el archivo de salida
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Definir el códec y crear el objeto VideoWriter para el archivo temporal
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Usar 'mp4v' para archivos .mp4
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        print(f"\nProcesando '{video_filename}'...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break # Fin del video
                
            # Lógica principal: eliminar los reflejos del fotograma
            clean_frame = remove_reflections(frame)
            
            # Escribir el fotograma limpio en el archivo de video temporal
            out.write(clean_frame)
            
            frame_count += 1
            # Imprimir una barra de progreso
            progress = int((frame_count / total_frames) * 50)
            sys.stdout.write(f"\r  [{'#' * progress}{'.' * (50 - progress)}] {frame_count}/{total_frames}")
            sys.stdout.flush()

        # Liberar los objetos de video
        cap.release()
        out.release()
        
        # --- PASO CRUCIAL: REEMPLAZAR EL ARCHIVO ORIGINAL ---
        try:
            os.remove(original_path) # Borrar el video original
            os.rename(temp_output_path, original_path) # Renombrar el temporal al nombre original
            print(f"\n✔️ Finalizado. El video '{original_path}' ha sido sobrescrito.")
        except Exception as e:
            print(f"\n❌ ERROR al reemplazar el archivo '{original_path}': {e}")
            print(f"   El archivo temporal se ha conservado como '{temp_output_path}'")

    print("\n\n🎉 ¡Todos los videos han sido procesados!")

if __name__ == "__main__":
    process_and_overwrite_videos(VIDEOS_FOLDER)
    cv2.destroyAllWindows()