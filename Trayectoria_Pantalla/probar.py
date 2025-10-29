import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
from collections import deque
import pickle
import tensorflow as tf
from tensorflow import keras
import glob

# Importar funciones de tu sistema de eye tracking
from utils.eye_tracker_utils import (
    crop_to_aspect_ratio,
    apply_fixed_binary_threshold,
    optimize_contours_by_angle,
    update_and_average_point,
    compute_average_intersection,
    compute_gaze_vector,
    FIXED_THRESHOLD_VALUE,
    GAUSSIAN_KERNEL_SIZE,
    CLAHE_CLIP_LIMIT,
    MORPH_KERNEL_SIZE,
    MIN_PUPIL_AREA,
    MAX_PUPIL_AREA,
    MIN_ELLIPTICAL_FIT_RATIO,
    MAX_ELLIPTICAL_FIT_RATIO,
    HORIZONTALITY_TOLERANCE,
    MAX_PUPIL_JUMP_DISTANCE,
    MAX_LOST_TRACK_FRAMES,
    MAX_INTERSECTION_DISTANCE
)

class GazeTrackingSystem:
    def __init__(self, model_path, metadata_path):
        # Cargar modelo y metadata
        print(f"\n=== Cargando modelo de IA ===")
        print(f"Modelo: {model_path}")
        print(f"Metadata: {metadata_path}")
        
        self.model = keras.models.load_model(model_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        self.screen_width = metadata['screen_width']
        self.screen_height = metadata['screen_height']
        
        print(f"✓ Modelo cargado correctamente")
        print(f"  - Resolución entrenada: {self.screen_width}x{self.screen_height}")
        
        # Configuración de ventana
        self.root = tk.Tk()
        self.root.title("Eye Tracking - Prueba en Tiempo Real")
        
        # Obtener dimensiones actuales de pantalla
        current_screen_width = self.root.winfo_screenwidth()
        current_screen_height = self.root.winfo_screenheight()
        
        # Configurar ventana fullscreen
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.8)
        self.root.configure(bg='black')
        
        # Canvas para dibujar
        self.canvas = tk.Canvas(
            self.root, 
            width=current_screen_width, 
            height=current_screen_height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Calcular factor de escala si la resolución cambió
        self.scale_x = current_screen_width / self.screen_width
        self.scale_y = current_screen_height / self.screen_height
        
        if self.scale_x != 1.0 or self.scale_y != 1.0:
            print(f"⚠ Ajustando escala de pantalla:")
            print(f"  - Resolución actual: {current_screen_width}x{current_screen_height}")
            print(f"  - Escala X: {self.scale_x:.3f}, Escala Y: {self.scale_y:.3f}")
        
        # Variables de estado
        self.gaze_position = np.array([current_screen_width / 2, current_screen_height / 2])
        self.smoothed_gaze = deque(maxlen=8)  # Suavizado temporal
        self.current_gaze_vector = None
        
        # Historial de posiciones para visualización
        self.gaze_trail = deque(maxlen=30)
        
        # Variables de eye tracking
        self.ray_lines = []
        self.model_centers = []
        self.stable_pupil_centers = []
        self.max_rays = 120
        self.prev_model_center_avg = (320, 240)
        self.max_observed_distance = 240
        self.last_known_pupil_center = None
        self.frames_since_last_good_detection = 0
        
        # Estadísticas
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Variables de video
        self.cap = None
        self.running = True
        self.video_thread = None
        
        # UI Elements
        self.red_dot = None
        self.info_text = None
        self.fps_text = None
        
        # Bind para salir
        self.root.bind('<Escape>', lambda e: self.quit())
        self.root.bind('r', lambda e: self.reset_tracking())
        self.root.bind('t', lambda e: self.toggle_trail())
        
        # Configuración de visualización
        self.show_trail = True
        
        # Dibujar UI inicial
        self.draw_ui()
    
    def draw_ui(self):
        """Dibuja los elementos de la interfaz"""
        # Punto rojo principal
        dot_size = 12
        x, y = self.gaze_position
        self.red_dot = self.canvas.create_oval(
            x - dot_size, y - dot_size,
            x + dot_size, y + dot_size,
            fill='red', outline='white', width=2
        )
        
        # Texto informativo
        info = "Eye Tracking con IA | ESC: salir | R: reset | T: toggle trail"
        self.info_text = self.canvas.create_text(
            self.canvas.winfo_reqwidth() / 2, 30,
            text=info,
            fill='white',
            font=('Arial', 12, 'bold')
        )
        
        # FPS counter
        self.fps_text = self.canvas.create_text(
            100, 60,
            text="FPS: --",
            fill='lime',
            font=('Arial', 10)
        )
    
    def toggle_trail(self):
        """Activa/desactiva el rastro de mirada"""
        self.show_trail = not self.show_trail
        if not self.show_trail:
            # Limpiar rastro
            for item in self.canvas.find_withtag("trail"):
                self.canvas.delete(item)
    
    def reset_tracking(self):
        """Resetea el sistema de tracking"""
        print("\nReseteando sistema de tracking...")
        self.ray_lines = []
        self.model_centers = []
        self.stable_pupil_centers = []
        self.prev_model_center_avg = (320, 240)
        self.last_known_pupil_center = None
        self.frames_since_last_good_detection = 0
        self.smoothed_gaze.clear()
        self.gaze_trail.clear()
    
    def process_eye_frame(self, frame):
        """Procesa un frame de video del ojo y extrae el vector de mirada"""
        # Preprocesamiento
        frame_processed = crop_to_aspect_ratio(frame)
        gray_frame = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
        gray_frame_blurred = cv2.GaussianBlur(gray_frame, GAUSSIAN_KERNEL_SIZE, 0)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        gray_frame_clahe = clahe.apply(gray_frame_blurred)
        
        # Binarización y morfología
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
        thresholded_image = apply_fixed_binary_threshold(gray_frame_clahe, FIXED_THRESHOLD_VALUE)
        thresholded_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        thresholded_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar por área
        contours_in_range = [c for c in contours if MIN_PUPIL_AREA <= cv2.contourArea(c) <= MAX_PUPIL_AREA]
        
        # Encontrar mejor contorno
        best_contour = None
        best_fit_score = float('inf')
        
        for contour in contours_in_range:
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w > (h * HORIZONTALITY_TOLERANCE):
                continue
            
            if len(contour) < 5:
                continue
            
            try:
                fitted_ellipse = cv2.fitEllipse(contour)
                width, height = fitted_ellipse[1]
                
                if width <= 0 or height <= 0:
                    continue
                
                ellipse_area = (np.pi / 4.0) * width * height
                contour_area = cv2.contourArea(contour)
                fit_ratio = contour_area / ellipse_area
                
                if MIN_ELLIPTICAL_FIT_RATIO < fit_ratio <= MAX_ELLIPTICAL_FIT_RATIO:
                    fit_score = abs(fit_ratio - 1.0)
                    if fit_score < best_fit_score:
                        best_fit_score = fit_score
                        best_contour = contour
            except cv2.error:
                continue
        
        # Procesar si encontramos un contorno válido
        if best_contour is not None:
            optimized_contour = optimize_contours_by_angle([best_contour])
            ellipse = None
            
            try:
                if len(optimized_contour) >= 5:
                    ellipse = cv2.fitEllipse(optimized_contour)
                elif len(best_contour) >= 5:
                    ellipse = cv2.fitEllipse(best_contour)
            except cv2.error:
                pass
            
            if ellipse is not None:
                center_x, center_y = map(int, ellipse[0])
                stable_center = update_and_average_point(self.stable_pupil_centers, (center_x, center_y), N=3)
                center_x, center_y = stable_center if stable_center else (center_x, center_y)
                
                # Filtro temporal
                new_pupil_center = (center_x, center_y)
                is_stable = False
                
                if self.last_known_pupil_center is None:
                    is_stable = True
                    self.last_known_pupil_center = new_pupil_center
                    self.frames_since_last_good_detection = 0
                else:
                    dist = np.hypot(
                        new_pupil_center[0] - self.last_known_pupil_center[0],
                        new_pupil_center[1] - self.last_known_pupil_center[1]
                    )
                    
                    if dist > MAX_PUPIL_JUMP_DISTANCE:
                        if self.frames_since_last_good_detection < MAX_LOST_TRACK_FRAMES:
                            is_stable = False
                            self.frames_since_last_good_detection += 1
                        else:
                            is_stable = True
                            self.last_known_pupil_center = new_pupil_center
                            self.frames_since_last_good_detection = 0
                    else:
                        is_stable = True
                        self.last_known_pupil_center = new_pupil_center
                        self.frames_since_last_good_detection = 0
                
                # Calcular centro del modelo
                model_center = compute_average_intersection(
                    frame_processed, self.ray_lines, 5, 1500, 5, self.prev_model_center_avg
                )
                
                if model_center is not None:
                    model_center_avg = update_and_average_point(self.model_centers, model_center, 800)
                    self.prev_model_center_avg = model_center_avg
                else:
                    model_center_avg = self.prev_model_center_avg
                
                # Si la detección es estable, calcular gaze
                if is_stable:
                    dist_from_center = np.hypot(
                        center_x - model_center_avg[0],
                        center_y - model_center_avg[1]
                    )
                    
                    if dist_from_center <= self.max_observed_distance:
                        self.ray_lines.append(ellipse)
                        if len(self.ray_lines) > self.max_rays:
                            self.ray_lines.pop(0)
                        
                        # Calcular vector de mirada 3D
                        center_3d, direction_3d = compute_gaze_vector(
                            center_x, center_y,
                            model_center_avg[0], model_center_avg[1],
                            self.max_observed_distance
                        )
                        
                        if center_3d is not None and direction_3d is not None:
                            # Dibujar en el frame de debug
                            cv2.ellipse(frame_processed, ellipse, (0, 255, 255), 2)
                            cv2.circle(frame_processed, model_center_avg, int(self.max_observed_distance), (255, 50, 50), 2)
                            cv2.line(frame_processed, model_center_avg, (center_x, center_y), (255, 150, 50), 2)
                            
                            # Mostrar vector de mirada
                            cv2.putText(frame_processed, f"Gaze: ({direction_3d[0]:.3f}, {direction_3d[1]:.3f}, {direction_3d[2]:.3f})",
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            return direction_3d, frame_processed
        
        return None, frame_processed
    
    def predict_screen_position(self, gaze_vector):
        """Predice la posición en pantalla usando el modelo de IA"""
        if gaze_vector is None:
            return None
        
        # Preparar entrada
        X = np.array([gaze_vector])
        X_scaled = self.scaler_X.transform(X)
        
        # Predecir
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0]
        
        # Aplicar escala si la resolución cambió
        screen_x = y_pred[0] * self.scale_x
        screen_y = y_pred[1] * self.scale_y
        
        # Clamp a límites de pantalla
        current_screen_width = self.root.winfo_screenwidth()
        current_screen_height = self.root.winfo_screenheight()
        
        screen_x = np.clip(screen_x, 0, current_screen_width)
        screen_y = np.clip(screen_y, 0, current_screen_height)
        
        return np.array([screen_x, screen_y])
    
    def update_gaze_position(self, new_position):
        """Actualiza la posición del punto con suavizado"""
        if new_position is None:
            return
        
        # Agregar a buffer de suavizado
        self.smoothed_gaze.append(new_position)
        
        # Calcular promedio ponderado (más peso a las muestras recientes)
        if len(self.smoothed_gaze) > 0:
            weights = np.linspace(0.5, 1.0, len(self.smoothed_gaze))
            weights = weights / weights.sum()
            
            positions = np.array(self.smoothed_gaze)
            self.gaze_position = np.average(positions, axis=0, weights=weights)
            
            # Agregar al rastro
            self.gaze_trail.append(self.gaze_position.copy())
    
    def video_processing_loop(self):
        """Loop principal de procesamiento de video"""
        cv2.namedWindow("Eye Tracking Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Eye Tracking Debug", 640, 480)
        
        while self.running:
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print("Error al leer frame de la cámara")
                time.sleep(0.1)
                continue
            
            # Procesar frame
            gaze_vector, debug_frame = self.process_eye_frame(frame)
            
            # Predecir posición en pantalla usando IA
            if gaze_vector is not None:
                self.current_gaze_vector = gaze_vector
                screen_pos = self.predict_screen_position(gaze_vector)
                self.update_gaze_position(screen_pos)
            
            # Calcular FPS
            frame_time = time.time() - frame_start
            self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
            
            # Mostrar frame de debug
            fps_avg = np.mean(self.fps_counter)
            cv2.putText(debug_frame, f"FPS: {fps_avg:.1f}", 
                       (10, debug_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Eye Tracking Debug", debug_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cv2.destroyAllWindows()
    
    def update_canvas(self):
        """Actualiza el canvas con la nueva posición del punto"""
        if not self.running:
            return
        
        # Dibujar rastro
        if self.show_trail and len(self.gaze_trail) > 1:
            # Limpiar rastro anterior
            for item in self.canvas.find_withtag("trail"):
                self.canvas.delete(item)
            
            # Dibujar nuevo rastro con degradado
            trail_points = list(self.gaze_trail)
            for i in range(len(trail_points) - 1):
                x1, y1 = trail_points[i]
                x2, y2 = trail_points[i + 1]
                
                # Color con opacidad basada en posición en el rastro
                alpha = int(255 * (i + 1) / len(trail_points))
                color = f'#{255:02x}{alpha:02x}{alpha:02x}'
                
                size = 2 + int(3 * (i + 1) / len(trail_points))
                
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill=color, width=size, tags="trail"
                )
        
        # Actualizar posición del punto rojo principal
        dot_size = 12
        x, y = self.gaze_position
        self.canvas.coords(
            self.red_dot,
            x - dot_size, y - dot_size,
            x + dot_size, y + dot_size
        )
        
        # Actualizar FPS
        if len(self.fps_counter) > 0:
            fps_avg = np.mean(self.fps_counter)
            self.canvas.itemconfig(self.fps_text, text=f"FPS: {fps_avg:.1f}")
        
        # Programar siguiente actualización
        self.root.after(16, self.update_canvas)  # ~60 FPS
    
    def start(self, camera_index=0):
        """Inicia el sistema de tracking"""
        # Abrir cámara
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {camera_index}")
            return
        
        print(f"✓ Cámara {camera_index} abierta correctamente")
        print("\n=== Sistema de Eye Tracking Activo ===")
        print("Controles:")
        print("  ESC - Salir")
        print("  R   - Resetear tracking")
        print("  T   - Activar/desactivar rastro")
        print("  Q   - Cerrar ventana de debug")
        print("\n¡El sistema está listo!")
        
        # Iniciar thread de procesamiento de video
        self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
        self.video_thread.start()
        
        # Iniciar actualización de canvas
        self.root.after(100, self.update_canvas)
        
        # Iniciar mainloop de Tkinter
        self.root.mainloop()
    
    def quit(self):
        """Cierra el programa limpiamente"""
        print("\n Cerrando sistema...")
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.root.quit()


def select_model():
    """Permite al usuario seleccionar un modelo guardado"""
    
    # --- INICIO DE LA CORRECCIÓN ---
    # Obtener la ruta absoluta del script que se está ejecutando
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construir la ruta de búsqueda de modelos relativa AL SCRIPT
    model_search_path = os.path.join(script_dir, 'models', 'gaze_model_*.h5')
    # --- FIN DE LA CORRECCIÓN ---

    model_files = glob.glob(model_search_path) # Usar la nueva ruta
    
    if not model_files:
        # Mensaje de error mejorado
        print(f"\n✗ Error: No se encontraron modelos en: {os.path.join(script_dir, 'models')}")
        print("  Ejecuta primero el script de entrenamiento")
        return None, None
    
    print("\n=== Modelos disponibles ===")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. {model_file}")
    
    if len(model_files) == 1:
        model_path = model_files[0]
        print(f"\n✓ Usando modelo: {model_path}")
    else:
        try:
            choice = int(input(f"\nSelecciona un modelo (1-{len(model_files)}): "))
            model_path = model_files[choice - 1]
        except (ValueError, IndexError):
            print("✗ Selección inválida")
            return None, None
    
    # Buscar archivo de metadata correspondiente
    metadata_path = model_path.replace('.h5', '_metadata.pkl')
    
    if not os.path.exists(metadata_path):
        print(f"✗ Error: No se encontró el archivo de metadata: {metadata_path}")
        return None, None
    
    return model_path, metadata_path


def main():
    print("=" * 60)
    print("  PRUEBA DE MODELO DE EYE TRACKING EN TIEMPO REAL")
    print("=" * 60)
    
    # Seleccionar modelo
    model_path, metadata_path = select_model()
    
    if model_path is None or metadata_path is None:
        return
    
    # Inicializar sistema
    try:
        system = GazeTrackingSystem(model_path, metadata_path)

        camera_index = 1
        
        # Iniciar
        system.start(camera_index=camera_index)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    main()