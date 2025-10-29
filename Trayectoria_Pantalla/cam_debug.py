import cv2
import numpy as np
from collections import deque
import time

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

class GazeVectorDebugger:
    def __init__(self):
        # Variables de eye tracking
        self.ray_lines = []
        self.model_centers = []
        self.stable_pupil_centers = []
        self.max_rays = 120
        self.prev_model_center_avg = (320, 240)
        self.max_observed_distance = 240
        self.last_known_pupil_center = None
        self.frames_since_last_good_detection = 0
        
        # Variables de debug
        self.current_gaze_vector = None
        self.gaze_history = deque(maxlen=100)  # Historial de vectores
        self.fps_counter = deque(maxlen=30)
        
        # Estado de detección
        self.detection_status = "No detectado"
        self.pupil_center = None
        self.sphere_center = None
        
    def process_eye_frame(self, frame):
        """Procesa un frame y extrae el vector de mirada"""
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
        
        # Resetear estado
        self.detection_status = "No detectado"
        gaze_vector = None
        
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
                            self.detection_status = "Inestable (salto detectado)"
                        else:
                            is_stable = True
                            self.last_known_pupil_center = new_pupil_center
                            self.frames_since_last_good_detection = 0
                            self.detection_status = "Reset (nuevo tracking)"
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
                
                self.sphere_center = model_center_avg
                
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
                            gaze_vector = direction_3d
                            self.pupil_center = (center_x, center_y)
                            self.detection_status = "✓ Detectado"
                            
                            # Dibujar en el frame
                            cv2.ellipse(frame_processed, ellipse, (0, 255, 255), 2)
                            cv2.circle(frame_processed, model_center_avg, int(self.max_observed_distance), (255, 50, 50), 2)
                            cv2.circle(frame_processed, model_center_avg, 8, (255, 255, 0), -1)
                            cv2.line(frame_processed, model_center_avg, (center_x, center_y), (255, 150, 50), 2)
                            
                            # Línea extendida de dirección
                            dx = center_x - model_center_avg[0]
                            dy = center_y - model_center_avg[1]
                            ex = int(model_center_avg[0] + 2 * dx)
                            ey = int(model_center_avg[1] + 2 * dy)
                            cv2.line(frame_processed, (center_x, center_y), (ex, ey), (200, 255, 0), 3)
                    else:
                        self.detection_status = "Fuera de rango"
        
        return gaze_vector, frame_processed
    
    def draw_vector_visualization(self, frame, gaze_vector):
        """Dibuja visualización del vector 3D en el frame"""
        h, w = frame.shape[:2]
        
        # Panel de información (fondo semi-transparente)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Título
        cv2.putText(frame, "=== GAZE VECTOR DEBUG ===", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Estado de detección
        color = (0, 255, 0) if "✓" in self.detection_status else (0, 165, 255)
        cv2.putText(frame, f"Estado: {self.detection_status}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Vector de mirada
        if gaze_vector is not None:
            cv2.putText(frame, f"Gaze Vector:", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"  X: {gaze_vector[0]:+.4f}", (10, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(frame, f"  Y: {gaze_vector[1]:+.4f}", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(frame, f"  Z: {gaze_vector[2]:+.4f}", (10, 165),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Magnitud del vector
            magnitude = np.linalg.norm(gaze_vector)
            cv2.putText(frame, f"Magnitude: {magnitude:.4f}", (10, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Gaze Vector: NONE", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Información adicional
        if self.pupil_center:
            cv2.putText(frame, f"Pupil: ({self.pupil_center[0]}, {self.pupil_center[1]})", 
                       (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        if self.sphere_center:
            cv2.putText(frame, f"Sphere: ({self.sphere_center[0]}, {self.sphere_center[1]})", 
                       (w - 250, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # FPS
        if len(self.fps_counter) > 0:
            fps = np.mean(self.fps_counter)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Historial (mini gráfica)
        if len(self.gaze_history) > 1:
            self.draw_vector_graph(frame, w, h)
        
        return frame
    
    def draw_vector_graph(self, frame, width, height):
        """Dibuja mini gráficas de la historia del vector"""
        graph_width = 300
        graph_height = 80
        graph_x = width - graph_width - 10
        graph_y = height - graph_height - 10
        
        # Fondo de gráfica
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Borde
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), 
                     (100, 100, 100), 1)
        
        # Título
        cv2.putText(frame, "Vector History", (graph_x + 5, graph_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Dibujar componentes X, Y, Z
        if len(self.gaze_history) > 1:
            history = np.array(list(self.gaze_history))
            n_points = len(history)
            
            # Normalizar para visualización
            center_y = graph_y + graph_height // 2
            scale = (graph_height // 2) * 0.8
            
            # Dibujar líneas
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=rojo, Y=verde, Z=azul
            labels = ['X', 'Y', 'Z']
            
            for component in range(3):
                points = []
                for i in range(n_points):
                    x = graph_x + int((i / max(n_points - 1, 1)) * (graph_width - 20)) + 10
                    value = history[i][component]
                    y = int(center_y - value * scale)
                    y = np.clip(y, graph_y + 20, graph_y + graph_height - 5)
                    points.append((x, y))
                
                # Dibujar línea
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], colors[component], 1)
                
                # Etiqueta
                cv2.putText(frame, labels[component], 
                           (graph_x + graph_width - 60 + component * 20, graph_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[component], 1)
            
            # Línea central (0)
            cv2.line(frame, (graph_x, center_y), 
                    (graph_x + graph_width, center_y), (100, 100, 100), 1)
    
    def run(self, camera_index=0):
        """Ejecuta el debugger en tiempo real"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"✗ Error: No se pudo abrir la cámara {camera_index}")
            return
        
        print("=" * 60)
        print("  GAZE VECTOR DEBUGGER")
        print("=" * 60)
        print(f"\n✓ Cámara {camera_index} abierta")
        print("\nControles:")
        print("  Q o ESC - Salir")
        print("  R       - Resetear tracking")
        print("  SPACE   - Pausar/Reanudar")
        print("\nInfo del Vector:")
        print("  X: Horizontal (- izquierda, + derecha)")
        print("  Y: Vertical   (- abajo, + arriba)")
        print("  Z: Profundidad (- hacia adelante)")
        print("\n¡Sistema iniciado!")
        
        paused = False
        
        # Crear ventana
        cv2.namedWindow("Gaze Vector Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gaze Vector Debug", 1280, 720)
        
        try:
            while True:
                frame_start = time.time()
                
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("✗ Error al leer frame")
                        break
                    
                    # Procesar frame
                    gaze_vector, processed_frame = self.process_eye_frame(frame)
                    
                    # Guardar en historial
                    if gaze_vector is not None:
                        self.current_gaze_vector = gaze_vector
                        self.gaze_history.append(gaze_vector)
                    
                    # Dibujar visualización
                    debug_frame = self.draw_vector_visualization(processed_frame, gaze_vector)
                    
                    # Calcular FPS
                    frame_time = time.time() - frame_start
                    if frame_time > 0:
                        self.fps_counter.append(1.0 / frame_time)
                else:
                    # Mostrar frame pausado
                    debug_frame = processed_frame.copy()
                    cv2.putText(debug_frame, "PAUSED", (debug_frame.shape[1] // 2 - 80, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                # Mostrar frame
                cv2.imshow("Gaze Vector Debug", debug_frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q o ESC
                    print("\n✓ Saliendo...")
                    break
                elif key == ord('r'):  # R - Reset
                    print("\n↻ Reseteando tracking...")
                    self.ray_lines = []
                    self.model_centers = []
                    self.stable_pupil_centers = []
                    self.prev_model_center_avg = (320, 240)
                    self.last_known_pupil_center = None
                    self.frames_since_last_good_detection = 0
                    self.gaze_history.clear()
                elif key == ord(' '):  # SPACE - Pausar
                    paused = not paused
                    print(f"\n{'⏸ Pausado' if paused else '▶ Reanudado'}")
        
        except KeyboardInterrupt:
            print("\n✓ Interrumpido por usuario")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Mostrar estadísticas finales
            if len(self.gaze_history) > 0:
                history = np.array(list(self.gaze_history))
                print("\n" + "=" * 60)
                print("  ESTADÍSTICAS FINALES")
                print("=" * 60)
                print(f"Vectores capturados: {len(history)}")
                print(f"\nPromedios:")
                print(f"  X: {np.mean(history[:, 0]):+.4f} ± {np.std(history[:, 0]):.4f}")
                print(f"  Y: {np.mean(history[:, 1]):+.4f} ± {np.std(history[:, 1]):.4f}")
                print(f"  Z: {np.mean(history[:, 2]):+.4f} ± {np.std(history[:, 2]):.4f}")
                print(f"\nRangos:")
                print(f"  X: [{np.min(history[:, 0]):+.4f}, {np.max(history[:, 0]):+.4f}]")
                print(f"  Y: [{np.min(history[:, 1]):+.4f}, {np.max(history[:, 1]):+.4f}]")
                print(f"  Z: [{np.min(history[:, 2]):+.4f}, {np.max(history[:, 2]):+.4f}]")


def main():

    camera_index = 1
    
    debugger = GazeVectorDebugger()
    debugger.run(camera_index=camera_index)


if __name__ == "__main__":
    main()