import cv2
import numpy as np
import os
import glob
import sys

# Dimensiones fijas de salida solicitadas
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480
OUTPUT_SIZE = (OUTPUT_WIDTH, OUTPUT_HEIGHT)

# Lista de extensiones de video que el programa intentará procesar
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv']

class ROISelector:
    """Clase para seleccionar una Región de Interés (ROI) y extraerla, 
       redimensionando el resultado a un tamaño fijo."""
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir el video en {video_path}")
            self.fps = 0
            self.width = 0
            self.height = 0
        else:
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        self.roi_selected = False
        self.roi_coordinates = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.fx, self.fy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.fx, self.fy = x, y
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            self.roi_selected = True
            
            x1, y1 = max(0, min(self.ix, self.fx)), max(0, min(self.iy, self.fy))
            x2, y2 = min(self.width, max(self.ix, self.fx)), min(self.height, max(self.iy, self.fy))
            self.roi_coordinates = (x1, y1, x2, y2)
            print(f"ROI seleccionada: ({x1}, {y1}) - ({x2}, {y2})")

    def select_roi(self):
        if not self.cap.isOpened():
            return False
            
        ret, frame = self.cap.read()
        if not ret:
            print(f"Error: No se pudo leer el primer frame del video {os.path.basename(self.video_path)}")
            return False
            
        window_name = f'Selecciona ROI para: {os.path.basename(self.video_path)}'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.roi_selected = False
        self.roi_coordinates = None
        self.drawing = False

        while True:
            display_frame = frame.copy()
            
            if self.drawing:
                 cv2.rectangle(display_frame, (self.ix, self.iy), (self.fx, self.fy), (0, 255, 0), 2)
            elif self.roi_selected and self.roi_coordinates:
                 x1, y1, x2, y2 = self.roi_coordinates
                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(display_frame, 'Click y arrastra para seleccionar ROI', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, 'Presiona ESPACIO para confirmar', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, 'Presiona ESC para saltar/cancelar', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                self.cap.release()
                cv2.destroyAllWindows()
                return False
            elif key == 32 and self.roi_selected:  # ESPACIO
                break
        
        cv2.destroyAllWindows()
        return True
    
    def extract_roi_video(self, output_path):
        if not self.cap.isOpened() or not self.roi_selected or self.roi_coordinates is None:
            print("Error: Video no abierto o ROI no seleccionada.")
            return False
        
        x1, y1, x2, y2 = self.roi_coordinates
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        if roi_width <= 0 or roi_height <= 0:
            print(f"Error: ROI inválida ({roi_width}x{roi_height}).")
            self.cap.release()
            return False
        
        # 🔑 CAMBIO 1: Configurar el video writer con las dimensiones fijas de salida
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, OUTPUT_SIZE) # Usar OUTPUT_SIZE
        
        if not out.isOpened():
             print(f"Error: No se pudo crear el VideoWriter en {output_path}. Verifica que el codec MP4V esté disponible.")
             self.cap.release()
             return False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Procesando video... Total frames: {total_frames}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 1. Extraer ROI del frame
            roi_frame = frame[y1:y2, x1:x2]
            
            # 🔑 CAMBIO 2: Redimensionar el frame de la ROI al tamaño fijo 640x480
            resized_frame = cv2.resize(roi_frame, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # 2. Escribir el frame redimensionado en el nuevo video
            out.write(resized_frame)
            
            # Mostrar progreso
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame % 100 == 0 or current_frame == total_frames:
                sys.stdout.write(f"  -> Procesando: {current_frame}/{total_frames} frames\r")
                sys.stdout.flush() 
        
        # Liberar recursos
        self.cap.release()
        out.release()
        
        print(f"\n✅ Video guardado exitosamente en: {output_path}")
        print(f"   Dimensiones del nuevo video: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}\n")
        return True

def main():
    # 1. Solicitar la ruta de la carpeta
    while True:
        folder_path = input("📂 Ingresa la ruta de la carpeta con los videos: ").strip().strip('"')
        if not os.path.isdir(folder_path):
            print("❌ Error: La ruta ingresada no es una carpeta válida. Intenta de nuevo.")
        else:
            break

    # 2. Configurar la carpeta de salida
    output_dir = os.path.join(folder_path, "ROI_videos_640x480")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Los videos con la ROI se guardarán en: {output_dir}")
    print(f"📐 Todos los videos de salida tendrán una resolución de {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}.\n")

    # 3. Obtener la lista de archivos de video
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not video_files:
        print(f"⚠️ No se encontraron archivos de video compatibles ({', '.join(VIDEO_EXTENSIONS)}) en la carpeta.")
        return

    # 4. Procesar cada video
    for i, video_path in enumerate(video_files):
        video_filename = os.path.basename(video_path)
        print("="*60)
        print(f"📹 [Video {i+1}/{len(video_files)}] Abriendo: {video_filename}")
        
        selector = ROISelector(video_path)
        
        if selector.select_roi():
            # Crear la ruta de salida con el sufijo "_ROI_640x480"
            base_name, ext = os.path.splitext(video_filename)
            output_filename = f"{base_name}_ROI_{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}{ext}"
            output_path = os.path.join(output_dir, output_filename)

            selector.extract_roi_video(output_path)
        else:
            print(f"⏭️ Selección de ROI cancelada/saltada para {video_filename}\n")

if __name__ == "__main__":
    # Importar sys para el manejo de la salida de progreso
    import sys
    main()