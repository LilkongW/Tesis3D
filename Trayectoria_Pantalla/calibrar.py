import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
from collections import deque
import pickle
from datetime import datetime

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

class EyeTrackingCalibrationSystem:
    def __init__(self):
        # Configuración de ventana
        self.root = tk.Tk()
        self.root.title("Eye Tracking Calibration System")
        
        # Obtener dimensiones de pantalla
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Configurar ventana fullscreen
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg='black')
        
        # Canvas para dibujar
        self.canvas = tk.Canvas(
            self.root, 
            width=self.screen_width, 
            height=self.screen_height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # === CALIBRACIÓN CON 9 PUNTOS ===
        self.calibration_points = self.generate_calibration_grid()
        self.current_calibration_index = 0
        self.calibration_duration = 3.0  # segundos por punto
        self.calibration_start_time = None
        self.is_calibrating = True
        self.calibration_data = []  # Lista de {gaze_vector: [x,y,z], screen_pos: [x,y]}
        
        # Variables de estado
        self.gaze_position = np.array([self.screen_width / 2, self.screen_height / 2])
        self.smoothed_gaze = deque(maxlen=10)
        self.current_gaze_vector = None
        
        # Variables de eye tracking
        self.ray_lines = []
        self.model_centers = []
        self.stable_pupil_centers = []
        self.max_rays = 120
        self.prev_model_center_avg = (320, 240)
        self.max_observed_distance = 240
        self.last_known_pupil_center = None
        self.frames_since_last_good_detection = 0
        
        # Variables de video
        self.cap = None
        self.running = True
        self.video_thread = None
        
        # UI Elements
        self.red_dot = None
        self.calibration_target = None
        self.info_text = None
        self.progress_text = None
        self.timer_text = None
        
        # Bind para salir
        self.root.bind('<Escape>', lambda e: self.quit())
        
        # Dibujar UI inicial
        self.draw_ui()
        
    def generate_calibration_grid(self):
        """Genera una grilla de 9 puntos para calibración (3x3)"""
        margin_x = self.screen_width * 0.15  # 15% de margen
        margin_y = self.screen_height * 0.15
        
        points = []
        for row in range(3):
            for col in range(3):
                x = margin_x + col * (self.screen_width - 2 * margin_x) / 2
                y = margin_y + row * (self.screen_height - 2 * margin_y) / 2
                points.append((int(x), int(y)))
        
        return points
    
    def draw_ui(self):
        """Dibuja los elementos de la interfaz"""
        # Limpiar canvas
        self.canvas.delete("all")
        
        if self.is_calibrating:
            # Dibujar punto de calibración actual
            if self.current_calibration_index < len(self.calibration_points):
                cx, cy = self.calibration_points[self.current_calibration_index]
                
                # Círculo pulsante grande
                size = 25
                self.calibration_target = self.canvas.create_oval(
                    cx - size, cy - size,
                    cx + size, cy + size,
                    fill='green', outline='white', width=3
                )
                
                # Punto central
                self.canvas.create_oval(
                    cx - 5, cy - 5,
                    cx + 5, cy + 5,
                    fill='white'
                )
                
                # Mostrar todos los puntos de la grilla (grises)
                for i, (px, py) in enumerate(self.calibration_points):
                    if i != self.current_calibration_index:
                        self.canvas.create_oval(
                            px - 8, py - 8,
                            px + 8, py + 8,
                            fill='gray', outline='darkgray'
                        )
                
                # Texto informativo
                info = f"CALIBRACIÓN: Mira el punto verde ({self.current_calibration_index + 1}/9)"
                self.info_text = self.canvas.create_text(
                    self.screen_width / 2, 50,
                    text=info,
                    fill='white',
                    font=('Arial', 18, 'bold')
                )
                
                # Texto de progreso
                if self.calibration_start_time:
                    elapsed = time.time() - self.calibration_start_time
                    remaining = max(0, self.calibration_duration - elapsed)
                    progress = f"Tiempo restante: {remaining:.1f}s"
                    self.progress_text = self.canvas.create_text(
                        self.screen_width / 2, 100,
                        text=progress,
                        fill='yellow',
                        font=('Arial', 16)
                    )
            else:
                # Calibración completada
                self.canvas.create_text(
                    self.screen_width / 2, self.screen_height / 2,
                    text="¡Calibración completada!\nGuardando datos...",
                    fill='green',
                    font=('Arial', 24, 'bold'),
                    justify='center'
                )
        else:
            # Modo de tracking activo
            # Punto rojo de seguimiento
            dot_size = 15
            x, y = self.gaze_position
            self.red_dot = self.canvas.create_oval(
                x - dot_size, y - dot_size,
                x + dot_size, y + dot_size,
                fill='red', outline='white', width=2
            )
            
            info = "Eye Tracking Activo | ESC: salir"
            self.info_text = self.canvas.create_text(
                self.screen_width / 2, 30,
                text=info,
                fill='white',
                font=('Arial', 14, 'bold')
            )
    
    def update_calibration(self):
        """Actualiza el proceso de calibración"""
        if not self.is_calibrating:
            return
        
        if self.current_calibration_index >= len(self.calibration_points):
            # Finalizar calibración
            self.finish_calibration()
            return
        
        # Iniciar timer si es necesario
        if self.calibration_start_time is None:
            self.calibration_start_time = time.time()
        
        elapsed = time.time() - self.calibration_start_time
        
        # Guardar datos de gaze si hay un vector válido
        if self.current_gaze_vector is not None:
            current_point = self.calibration_points[self.current_calibration_index]
            
            # Guardar muestra
            self.calibration_data.append({
                'gaze_vector': self.current_gaze_vector.copy(),
                'screen_pos': np.array(current_point, dtype=np.float32),
                'timestamp': time.time(),
                'point_index': self.current_calibration_index
            })
        
        # Verificar si completamos este punto
        if elapsed >= self.calibration_duration:
            print(f"Punto {self.current_calibration_index + 1}/9 completado - "
                  f"Muestras recolectadas: {len([d for d in self.calibration_data if d['point_index'] == self.current_calibration_index])}")
            
            # Avanzar al siguiente punto
            self.current_calibration_index += 1
            self.calibration_start_time = None
            
            # Redibujar UI
            self.draw_ui()
        else:
            # Actualizar progreso
            self.draw_ui()
        
        # Programar siguiente actualización
        if self.running:
            self.root.after(50, self.update_calibration)  # 20 FPS
    
    def finish_calibration(self):
        """Finaliza la calibración y guarda los datos"""
        self.is_calibrating = False
        
        print("\n=== CALIBRACIÓN COMPLETADA ===")
        print(f"Total de muestras recolectadas: {len(self.calibration_data)}")
        
        # Analizar datos por punto
        for i in range(9):
            samples = [d for d in self.calibration_data if d['point_index'] == i]
            print(f"Punto {i+1}: {len(samples)} muestras")
        
        # Guardar en archivo .pkl
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/vit/Documentos/Tesis3D/Trayectoria_Pantalla/calibracion_data/calibration_data_{timestamp}.pkl"
        
        calibration_package = {
            'calibration_points': self.calibration_points,
            'calibration_data': self.calibration_data,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'timestamp': timestamp
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(calibration_package, f)
            print(f"\n✓ Datos guardados en: {filename}")
            print(f"  - {len(self.calibration_data)} muestras totales")
            print("  - 9 puntos de calibración")
            print(f"  - Resolución: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            print(f"✗ Error al guardar: {e}")
        
        # Redibujar UI
        self.draw_ui()
        
        # Esperar 2 segundos y luego activar tracking
        self.root.after(2000, self.activate_tracking)
    
    def activate_tracking(self):
        """Activa el modo de tracking después de calibración"""
        self.draw_ui()
        self.root.after(16, self.update_canvas)
    
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
    
    def gaze_to_screen_position(self, gaze_direction):
        """Convierte el vector de mirada 3D a posición en pantalla (método simple)"""
        if gaze_direction is None:
            return None
        
        gaze_x, gaze_y, gaze_z = gaze_direction
        
        # Mapeo simple (se mejorará con el modelo de IA)
        sensitivity = 800
        
        screen_x = (self.screen_width / 2) - (gaze_x * sensitivity)
        screen_y = (self.screen_height / 2) - (gaze_y * sensitivity)
        
        screen_x = np.clip(screen_x, 0, self.screen_width)
        screen_y = np.clip(screen_y, 0, self.screen_height)
        
        return np.array([screen_x, screen_y])
    
    def update_gaze_position(self, new_position):
        """Actualiza la posición del punto rojo con suavizado"""
        if new_position is None:
            return
        
        self.smoothed_gaze.append(new_position)
        
        if len(self.smoothed_gaze) > 0:
            self.gaze_position = np.mean(self.smoothed_gaze, axis=0)
    
    def video_processing_loop(self):
        """Loop principal de procesamiento de video"""
        cv2.namedWindow("Eye Tracking Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Eye Tracking Debug", 640, 480)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error al leer frame de la cámara")
                time.sleep(0.1)
                continue
            
            # Procesar frame
            gaze_direction, debug_frame = self.process_eye_frame(frame)
            
            # Actualizar vector de mirada actual
            if gaze_direction is not None:
                self.current_gaze_vector = gaze_direction
                
                # Si no estamos calibrando, actualizar posición en pantalla
                if not self.is_calibrating:
                    screen_pos = self.gaze_to_screen_position(gaze_direction)
                    self.update_gaze_position(screen_pos)
            
            # Mostrar frame de debug
            cv2.imshow("Eye Tracking Debug", debug_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cv2.destroyAllWindows()
    
    def update_canvas(self):
        """Actualiza el canvas con la nueva posición del punto"""
        if not self.running or self.is_calibrating:
            return
        
        # Actualizar posición del punto rojo
        if self.red_dot:
            dot_size = 15
            x, y = self.gaze_position
            self.canvas.coords(
                self.red_dot,
                x - dot_size, y - dot_size,
                x + dot_size, y + dot_size
            )
        
        # Programar siguiente actualización
        self.root.after(16, self.update_canvas)
    
    def start(self, camera_index=0):
        """Inicia el sistema de eye tracking"""
        # Abrir cámara
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {camera_index}")
            return
        
        print(f"Cámara {camera_index} abierta correctamente")
        print("\n=== SISTEMA DE CALIBRACIÓN ===")
        print("Vas a mirar 9 puntos en la pantalla durante 3 segundos cada uno")
        print("Los datos se guardarán en un archivo .pkl para entrenar el modelo de IA")
        print("\nInstrucciones:")
        print("  1. Mantén la cabeza quieta durante toda la calibración")
        print("  2. Mira fijamente cada punto verde cuando aparezca")
        print("  3. El sistema recolectará datos automáticamente")
        print("  4. Presiona ESC para cancelar en cualquier momento")
        print("\n¡La calibración comenzará en 3 segundos!")
        
        # Iniciar thread de procesamiento de video
        self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
        self.video_thread.start()
        
        # Iniciar calibración después de 3 segundos
        self.root.after(3000, self.update_calibration)
        
        # Iniciar mainloop de Tkinter
        self.root.mainloop()
    
    def quit(self):
        """Cierra el programa limpiamente"""
        print("\nCerrando...")
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    try:
        system = EyeTrackingCalibrationSystem()
        system.start(camera_index=1)  # Cambia a 0 si es necesario
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()