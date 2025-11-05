import os
import random
import cv2

# Configuración de rutas (modifícalas según tus necesidades)
CARPETA_VIDEOS = r"C:\Users\Victor\Documents\Tesis3D\Videos\Experimento_1"
CARPETA_SALIDA = r"C:\Users\Victor\Documents\Tesis3D\Presentacion_imagenes\frames"

def extraer_frame(ruta_video, numero_frame):
    """
    Extrae un frame específico de un video
    """
    try:
        # Abrir el video
        cap = cv2.VideoCapture(ruta_video)
        
        # Verificar si el video se abrió correctamente
        if not cap.isOpened():
            print(f"No se pudo abrir el video: {ruta_video}")
            return None
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video: {os.path.basename(ruta_video)}")
        print(f"  - FPS: {fps:.2f}")
        print(f"  - Total frames: {total_frames}")
        
        # Verificar si el frame solicitado existe
        if numero_frame >= total_frames:
            print(f"  ⚠️  El frame {numero_frame} no existe. Usando el último frame disponible.")
            numero_frame = total_frames - 1
        
        # Posicionar en el frame deseado
        cap.set(cv2.CAP_PROP_POS_FRAMES, numero_frame)
        
        # Leer el frame
        ret, frame = cap.read()
        
        # Liberar el video
        cap.release()
        
        if ret:
            print(f"  ✅ Frame {numero_frame} extraído correctamente")
            return frame
        else:
            print(f"  ❌ No se pudo leer el frame {numero_frame}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error al procesar el video: {str(e)}")
        return None

def main():
    """
    Función principal que busca videos en todas las subcarpetas y extrae el frame 60
    """
    print("=" * 60)
    print("EXTRACTOR DE FRAMES - TESIS 3D")
    print("=" * 60)
    print(f"Carpeta de videos: {CARPETA_VIDEOS}")
    print(f"Carpeta de salida: {CARPETA_SALIDA}")
    print("-" * 60)
    
    # Verificar que la carpeta de videos existe
    if not os.path.exists(CARPETA_VIDEOS):
        print(f"❌ ERROR: La carpeta de videos no existe: {CARPETA_VIDEOS}")
        return
    
    # Crear carpeta de salida si no existe
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    print(f"✅ Carpeta de salida creada/verificada: {CARPETA_SALIDA}")
    print("-" * 60)
    
    # Contadores para estadísticas
    total_carpetas = 0
    total_videos_procesados = 0
    total_frames_guardados = 0
    
    # Recorrer todas las subcarpetas
    for root, dirs, files in os.walk(CARPETA_VIDEOS):
        # Buscar archivos de video en la carpeta actual
        videos = []
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
                videos.append(os.path.join(root, file))
        
        # Si hay videos en esta carpeta
        if videos:
            total_carpetas += 1
            print(f"\n📁 Carpeta: {os.path.basename(root)}")
            print(f"  Encontrados {len(videos)} videos")
            
            # Seleccionar un video aleatorio
            video_seleccionado = random.choice(videos)
            
            # Extraer el frame 60
            frame_extraido = extraer_frame(video_seleccionado, 60)
            
            if frame_extraido is not None:
                total_videos_procesados += 1
                
                # Crear nombre para el frame
                nombre_carpeta = os.path.basename(root)
                if not nombre_carpeta or nombre_carpeta == "Experimento_1":
                    nombre_carpeta = "raiz"
                
                # Limpiar el nombre de caracteres no válidos
                nombre_limpio = "".join(c for c in nombre_carpeta if c.isalnum() or c in (' ', '-', '_')).rstrip()
                
                # Guardar el frame
                nombre_salida = f"frame_60_{nombre_limpio}.jpg"
                ruta_completa = os.path.join(CARPETA_SALIDA, nombre_salida)
                
                # Si ya existe un archivo con ese nombre, agregar un número
                contador = 1
                ruta_original = ruta_completa
                while os.path.exists(ruta_completa):
                    nombre, ext = os.path.splitext(ruta_original)
                    ruta_completa = f"{nombre}_{contador}{ext}"
                    contador += 1
                
                cv2.imwrite(ruta_completa, frame_extraido)
                total_frames_guardados += 1
                print(f"  💾 Frame guardado: {os.path.basename(ruta_completa)}")
            else:
                print(f"  ❌ No se pudo extraer frame del video seleccionado")
    
    # Mostrar resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DEL PROCESO")
    print("=" * 60)
    print(f"📂 Carpetas procesadas: {total_carpetas}")
    print(f"🎥 Videos procesados: {total_videos_procesados}")
    print(f"🖼️  Frames guardados: {total_frames_guardados}")
    print(f"📁 Frames guardados en: {CARPETA_SALIDA}")
    print("=" * 60)

if __name__ == "__main__":
    main()