# main_tracker_integrated.py

import cv2
import numpy as np
import os
import sys
import threading
import time
import math
import random

from gl_sphere import (
    start_gl_window, 
    update_sphere_rotation_signal,
    get_latest_rendered_image,
    app, 
    sphere_widget,
    gl_image_buffer
)

# =================================================================
# VARIABLES GLOBALES Y PARÁMETROS DE DETECCIÓN
# =================================================================

# --- PARÁMETROS DE FILTRADO DE CONTORNOS ---
FIXED_THRESHOLD_VALUE = 90  # Umbral fijo para binarización
GAUSSIAN_KERNEL_SIZE = (7, 7) 
MIN_PUPIL_AREA = 1200  
MAX_PUPIL_AREA = 10000 
MAX_ELLIPSE_RATIO = 2.2 
# --- PARÁMETRO DE ESTABILIDAD DEL MODELO ---
MAX_INTERSECTION_DISTANCE = 100 # Distancia máxima en píxeles que una intersección puede estar del centro del modelo previo.

# --- VARIABLES DE ESTADO ---
ray_lines = [] 
model_centers = []
stable_pupil_centers = [] # Para suavizar la posición de la pupila
max_rays = 100
prev_model_center_avg = (320, 240)
max_observed_distance = 230 # Radio fijo del círculo cian

# Variables para la comunicación entre hilos
gl_ready = threading.Event()

# =================================================================
# --- FUNCIONES DE LÓGICA DE DETECCIÓN (INTEGRADAS) ---
# =================================================================

def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height
    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]
    return cv2.resize(cropped_img, (width, height))

def apply_fixed_binary_threshold(image, threshold_value):
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

def optimize_contours_by_angle(contours):
    if len(contours) < 1 or len(contours[0]) < 5:
        return np.array([], dtype=np.int32).reshape((-1, 1, 2))
    all_contours = np.concatenate(contours[0], axis=0)
    filtered_points = []
    centroid = np.mean(all_contours, axis=0)
    for i in range(0, len(all_contours), 1):
        current_point = all_contours[i]
        prev_point = all_contours[i - 1]
        next_point = all_contours[(i + 1) % len(all_contours)]
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        with np.errstate(invalid='ignore'):
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        vec_to_centroid = centroid - current_point
        cos_threshold = np.cos(np.radians(60))
        if np.dot(vec_to_centroid, (vec1 + vec2) / 2) >= cos_threshold:
            filtered_points.append(current_point)
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

def detect_and_select_pupil(gray_frame, threshold_value):
    thresholded_image = apply_fixed_binary_threshold(gray_frame, threshold_value)
    contours, _ = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_pupil_contour = None
    min_average_darkness = 256
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if not (MIN_PUPIL_AREA <= contour_area <= MAX_PUPIL_AREA):
            continue
        if len(contour) < 5:
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            if minor_axis == 0: continue
            aspect_ratio = major_axis / minor_axis
            if aspect_ratio <= MAX_ELLIPSE_RATIO:
                mask = np.zeros_like(gray_frame, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                dark_pixels = gray_frame[mask == 255]
                if dark_pixels.size > 0:
                    average_darkness = np.mean(dark_pixels)
                else:
                    continue
                if average_darkness < min_average_darkness:
                    min_average_darkness = average_darkness
                    best_pupil_contour = contour
        except Exception:
            continue
    return best_pupil_contour

def update_and_average_point(point_list, new_point, N):
    point_list.append(new_point)
    if len(point_list) > N:
        point_list.pop(0)
    if not point_list:
        return None
    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))
    return (avg_x, avg_y)

def find_line_intersection(ellipse1, ellipse2):
    (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
    (cx2, cy2), (_, minor_axis2), angle2 = ellipse2
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)
    dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
    dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([cx2 - cx1, cy2 - cy1])
    if np.linalg.det(A) == 0:
        return None
    try:
        t1, t2 = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1
    return (int(intersection_x), int(intersection_y))

def compute_average_intersection(frame, ray_lines, N, M, spacing):
    global stored_intersections, prev_model_center_avg
    stored_intersections = getattr(compute_average_intersection, 'stored_intersections', [])
    if len(ray_lines) < 2 or N < 2:
        return None
    height, width = frame.shape[:2]
    selected_lines = random.sample(ray_lines, min(N, len(ray_lines)))
    intersections = []
    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]
        if abs(line1[2] - line2[2]) >= 2:
            intersection = find_line_intersection(line1, line2)
            if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                dist = math.hypot(intersection[0] - prev_model_center_avg[0], intersection[1] - prev_model_center_avg[1])
                if dist < MAX_INTERSECTION_DISTANCE:
                    intersections.append(intersection)
                    stored_intersections.append(intersection)
    if len(stored_intersections) > M:
        stored_intersections = stored_intersections[-M:]
    compute_average_intersection.stored_intersections = stored_intersections
    if not stored_intersections:
        return None
    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])
    return (int(avg_x), int(avg_y))

# =================================================================
# THREAD DE VISUALIZACIÓN 3D
# =================================================================
def run_gl_thread():
    """Inicia la aplicación PyQt/OpenGL en un hilo separado."""
    global app
    app = start_gl_window()
    gl_ready.set()
    sys.exit(app.exec_())

# =================================================================
# CÓDIGO DE PROCESAMIENTO (AHORA USA LA LÓGICA INTEGRADA)
# =================================================================
def process_frame_for_comparison(frame):
    """Procesa un frame usando la lógica robusta y devuelve datos para visualización."""
    global ray_lines, model_centers, max_rays, prev_model_center_avg, max_observed_distance, stable_pupil_centers
    
    # 1. Pre-procesamiento
    frame = crop_to_aspect_ratio(frame)
    frame_orig_shape = frame.shape
    processed_frame = cv2.GaussianBlur(frame, GAUSSIAN_KERNEL_SIZE, 0)
    gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Detección principal de pupila
    pupil_contour = detect_and_select_pupil(gray_frame, FIXED_THRESHOLD_VALUE)
    
    final_rotated_rect = None
    center_x, center_y = None, None
    pupil_center_smoothed = None

    if pupil_contour is not None and len(pupil_contour) >= 5:
        # Optimizar contorno y ajustar elipse
        optimized_contour = optimize_contours_by_angle([pupil_contour])
        if len(optimized_contour) >= 5:
            ellipse = cv2.fitEllipse(optimized_contour)
            final_rotated_rect = ellipse
            center_x, center_y = map(int, final_rotated_rect[0])
            
            # Suavizar la posición de la pupila
            pupil_center_smoothed = update_and_average_point(stable_pupil_centers, (center_x, center_y), N=5)
            
            # Almacenar el rayo (elipse) para el cálculo del centro del modelo
            ray_lines.append(final_rotated_rect)
            if len(ray_lines) > max_rays:
                ray_lines = ray_lines[-max_rays:]
    
    # 3. Calcular centro del modelo (esfera cian)
    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5)
    
    # Suavizar el centro del modelo
    if model_center is not None:
        model_center_average = update_and_average_point(model_centers, model_center, 800)
        prev_model_center_avg = model_center_average
    else:
        model_center_average = prev_model_center_avg
    
    # 4. Dibujar en el frame original
    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)
    
    if final_rotated_rect is not None and pupil_center_smoothed is not None:
        px, py = pupil_center_smoothed
        cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2)
        cv2.line(frame, model_center_average, (px, py), (255, 150, 50), 2)
        
        # Línea extendida
        dx = px - model_center_average[0]
        dy = py - model_center_average[1]
        extended_x = int(model_center_average[0] + 2 * dx)
        extended_y = int(model_center_average[1] + 2 * dy)
        cv2.line(frame, (px, py), (extended_x, extended_y), (200, 255, 0), 3)

    return frame, model_center_average, pupil_center_smoothed, frame_orig_shape

# =================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO DE VIDEO
# =================================================================
def process_video_comparison(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video '{video_path}' no existe.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video '{video_path}'.")
        return
    
    gl_thread = threading.Thread(target=run_gl_thread)
    gl_thread.start()
    
    print("⏳ Esperando que la ventana 3D de OpenGL se inicie...")
    gl_ready.wait()
    print("✅ Ventana 3D lista.")

    print(f"Procesando: {video_path}")
    print("\nControles:\n   'q' - Salir\n   'ESPACIO' - Pausar/Reanudar\n" + "-"*60)
    
    frame_number = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n⏹️  Fin del video alcanzado")
                    break
                
                try:
                    # 1. Procesar frame y obtener los datos
                    processed_frame, model_center, pupil_center, frame_shape = \
                        process_frame_for_comparison(frame)
                    
                    # 2. ENVIAR los datos al hilo de OpenGL
                    if pupil_center and model_center:
                        update_sphere_rotation_signal(
                            pupil_center[0], pupil_center[1], 
                            model_center[0], model_center[1], 
                            frame_shape[1], frame_shape[0]
                        )
                    
                    # 3. RECUPERAR la última imagen renderizada de OpenGL
                    gl_image = get_latest_rendered_image()
                        
                    # 4. COMBINAR AMBAS IMÁGENES
                    if gl_image is not None:
                        clean_view = cv2.cvtColor(gl_image, cv2.COLOR_RGB2BGR)
                        h_cv, w_cv, _ = processed_frame.shape
                        h_gl, w_gl, _ = clean_view.shape

                        if h_cv != h_gl:
                            new_w_gl = int(w_gl * h_cv / h_gl)
                            clean_view = cv2.resize(clean_view, (new_w_gl, h_cv), interpolation=cv2.INTER_LINEAR)
                            
                        combined_frame = np.hstack((processed_frame, clean_view))
                        cv2.imshow("Eye Tracking: Real vs. 3D Model", combined_frame)
                    else:
                        cv2.imshow("Eye Tracking: Real vs. 3D Model", processed_frame)
                    
                    frame_number += 1
                    if frame_number % 30 == 0:
                        print(f"📊 Frame {frame_number} procesado", end='\r')
                            
                except Exception as e:
                    print(f"\n⚠️  Error procesando frame {frame_number}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                print("\n🛑 Detenido por usuario")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"\n{'⏸️  PAUSADO' if paused else '▶️  REPRODUCIENDO'}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if app is not None:
            app.quit()
        if gl_thread.is_alive():
             gl_thread.join(timeout=1)
        print("\n✅ Procesamiento completado")
        print(f"📊 Total frames: {frame_number}")

def main():
    print("="*60 + "\nCOMPARADOR DE EYE TRACKING - OJO REAL VS VIRTUAL\n" + "="*60)
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = r"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/Victor/ROI_videos_640x480/grabacion_experimento_ESP32CAM_1_ROI_640x480.mp4"
    
    process_video_comparison(video_path)

if __name__ == "__main__":
    main()