# main_tracker_integrated.py
# VERSIÓN DEMO CON GUARDADO DE VIDEO
# ¡AHORA ENVÍA EL TAMAÑO DE LA PUPILA AL 3D!

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
# VARIABLES GLOBALES Y PARÁMETROS DE DETECCIÓN (Sin cambios)
# =================================================================
FIXED_THRESHOLD_VALUE = 30
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0
MIN_PUPIL_AREA = 1000
MAX_PUPIL_AREA = 8000
MORPH_KERNEL_SIZE = 5
MAX_INTERSECTION_DISTANCE = 95 
MIN_ELLIPTICAL_FIT_RATIO = 0.8
MAX_ELLIPTICAL_FIT_RATIO = 1.20
HORIZONTALITY_TOLERANCE = 1.30
MAX_PUPIL_JUMP_DISTANCE = 40
MAX_LOST_TRACK_FRAMES = 10
ray_lines = []
model_centers = []
stable_pupil_centers = []
max_rays = 120
prev_model_center_avg = (320, 240)
max_observed_distance = 240 
last_known_pupil_center = None
frames_since_last_good_detection = 0
gl_ready = threading.Event()
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))


# =================================================================
# --- FUNCIONES DE LÓGICA DE DETECCIÓN (Modificadas) ---
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
    if not isinstance(contours, list) or len(contours) < 1 or len(contours[0]) < 5: return np.array([], dtype=np.int32).reshape((-1, 1, 2))
    if len(contours[0].shape) == 2: all_contours = contours[0].reshape((-1, 1, 2))
    else: all_contours = contours[0]
    spacing = max(1, int(len(all_contours)/25)); filtered_points = []
    centroid = np.mean(all_contours, axis=0).reshape(2)
    for i in range(len(all_contours)):
        current_point = all_contours[i].reshape(2)
        prev_point = all_contours[i - spacing].reshape(2)
        next_point = all_contours[(i + spacing) % len(all_contours)].reshape(2)
        vec1 = prev_point - current_point; vec2 = next_point - current_point
        with np.errstate(invalid='ignore'):
            norm_vec1 = np.linalg.norm(vec1); norm_vec2 = np.linalg.norm(vec2)
            if norm_vec1 == 0 or norm_vec2 == 0: continue
            dot_product = np.dot(vec1, vec2)
            dot_product = np.clip(dot_product / (norm_vec1 * norm_vec2), -1.0, 1.0)
        vec_to_centroid = centroid - current_point
        if np.dot(vec_to_centroid, (vec1+vec2)) > 0: filtered_points.append(all_contours[i])
    if not filtered_points or len(filtered_points) < 5: return np.array([], dtype=np.int32).reshape((-1, 1, 2))
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))


def process_frames(frame, gray_frame_clahe):
    """Función de lógica principal de eye_tracker_utils.py"""
    global ray_lines, max_rays, prev_model_center_avg, max_observed_distance, stable_pupil_centers, model_centers
    global last_known_pupil_center, frames_since_last_good_detection
    global morph_kernel 

    # 1. Binarización y Morfología
    thresholded_image_raw = apply_fixed_binary_threshold(gray_frame_clahe, FIXED_THRESHOLD_VALUE)
    thresholded_image_closed = cv2.morphologyEx(thresholded_image_raw, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    thresholded_image_final = cv2.morphologyEx(thresholded_image_closed, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # 2. Encontrar Contornos
    contours, _ = cv2.findContours(thresholded_image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Pre-filtrar por Área
    contours_in_area_range = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if MIN_PUPIL_AREA <= contour_area <= MAX_PUPIL_AREA:
            contours_in_area_range.append(contour)

    # 4. Encontrar el Mejor Contorno
    best_pupil_contour = None
    best_fit_score = float('inf') 

    for contour in contours_in_area_range:
        x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(contour)
        if h_bbox == 0: continue 
        if w_bbox > (h_bbox * HORIZONTALITY_TOLERANCE):
            continue
        if len(contour) < 5: continue
        try:
            fitted_ellipse = cv2.fitEllipse(contour)
            (width, height) = fitted_ellipse[1]
            if width <= 0 or height <= 0: continue
            ellipse_area = (np.pi / 4.0) * width * height
            if ellipse_area <= 0: continue
            contour_area = cv2.contourArea(contour)
            fit_ratio = contour_area / ellipse_area
            if MIN_ELLIPTICAL_FIT_RATIO < fit_ratio <= MAX_ELLIPTICAL_FIT_RATIO:
                current_fit_score = abs(fit_ratio - 1.0) 
                if current_fit_score < best_fit_score:
                    best_fit_score = current_fit_score
                    best_pupil_contour = contour
        except cv2.error:
            continue
    
    # 5. Procesar si se encontró un contorno válido
    final_rotated_rect = None
    center_x, center_y = None, None
    is_detection_temporally_stable = False
    valid_pupil_for_3d = False 
    scaled_pupil_radius = 0.0 # --- ¡AÑADIDO! Valor por defecto

    if best_pupil_contour is not None: 
        optimized_contour = optimize_contours_by_angle([best_pupil_contour])
        ellipse = None
        try:
            if len(optimized_contour) >= 5: ellipse = cv2.fitEllipse(optimized_contour)
            elif len(best_pupil_contour) >= 5: ellipse = cv2.fitEllipse(best_pupil_contour)
        except cv2.error: ellipse = None

        if ellipse is not None:
            final_rotated_rect = ellipse
            center_x_raw, center_y_raw = map(int, final_rotated_rect[0])
            stable_pupil_center = update_and_average_point(stable_pupil_centers, (center_x_raw, center_y_raw), N=3)
            center_x, center_y = stable_pupil_center if stable_pupil_center else (center_x_raw, center_y_raw)
            
            new_pupil_center = (center_x, center_y)
            if last_known_pupil_center is None:
                is_detection_temporally_stable = True
                last_known_pupil_center = new_pupil_center
                frames_since_last_good_detection = 0
            else:
                dist = math.hypot(new_pupil_center[0] - last_known_pupil_center[0], 
                                 new_pupil_center[1] - last_known_pupil_center[1])
                if dist > MAX_PUPIL_JUMP_DISTANCE:
                    if frames_since_last_good_detection < MAX_LOST_TRACK_FRAMES:
                        is_detection_temporally_stable = False
                        frames_since_last_good_detection += 1
                    else:
                        is_detection_temporally_stable = True
                        last_known_pupil_center = new_pupil_center
                        frames_since_last_good_detection = 0
                else:
                    is_detection_temporally_stable = True
                    last_known_pupil_center = new_pupil_center
                    frames_since_last_good_detection = 0
    else:
        frames_since_last_good_detection += 1

    # Calcular centro del modelo (Esfera Cian)
    model_center_average = prev_model_center_avg
    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5, model_center_average)
    if model_center is not None:
        model_center_average = update_and_average_point(model_centers, model_center, 800)
        prev_model_center_avg = model_center_average
    
    if is_detection_temporally_stable: 
        dist_from_sphere_center = math.hypot(center_x - model_center_average[0], 
                                             center_y - model_center_average[1])
        if dist_from_sphere_center <= max_observed_distance:
            
            valid_pupil_for_3d = True 
            ray_lines.append(final_rotated_rect) 
            if len(ray_lines) > max_rays: ray_lines.pop(0)

            # --- ¡LÓGICA DE TAMAÑO DE PUPILA AÑADIDA! ---
            ellipse_width = final_rotated_rect[1][0]
            ellipse_height = final_rotated_rect[1][1]
            avg_radius_pixels = (ellipse_width + ellipse_height) / 4.0 # 2 por promedio, 2 por radio
            
            # Escalar el radio de píxeles a unidades GL (donde 1.0 es el radio del globo ocular)
            # Añadir un fallback (0.001) para evitar división por cero
            scaled_pupil_radius = (avg_radius_pixels / (max_observed_distance + 0.001))
            
            # Limitar el radio de la pupila para que no sea más grande que el iris (ej. 0.39)
            scaled_pupil_radius = min(scaled_pupil_radius, 0.39)
            # --- FIN DE LA LÓGICA DE TAMAÑO ---

            # Dibujar
            cv2.ellipse(frame, final_rotated_rect, (0, 255, 255), 2) # Amarillo
            cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2) # Azul claro
            dx = center_x - model_center_average[0]; dy = center_y - model_center_average[1]
            ex = int(model_center_average[0] + 2 * dx); ey = int(model_center_average[1] + 2 * dy)
            cv2.line(frame, (center_x, center_y), (ex, ey), (200, 255, 0), 3) # Verde lima
        
    # Dibujar modelo del ojo (siempre)
    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2) # Azul oscuro
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1) # Cian

    pupil_center_final = (center_x, center_y) if valid_pupil_for_3d else None
    
    # --- ¡MODIFICADO! Devolver el radio escalado ---
    # Si no es válida, devuelve 0.0, que se manejará en el script 3D
    scaled_radius_final = scaled_pupil_radius if valid_pupil_for_3d else 0.0
    
    return frame, model_center_average, pupil_center_final, scaled_radius_final


def update_and_average_point(point_list, new_point, N):
    point_list.append(new_point)
    if len(point_list) > N: point_list.pop(0)
    if not point_list: return None
    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))
    return (avg_x, avg_y)

def compute_average_intersection(frame, ray_lines, N, M, spacing, current_center_avg):
    if not hasattr(compute_average_intersection, 'stored_intersections'):
        compute_average_intersection.stored_intersections = []
    stored_intersections = compute_average_intersection.stored_intersections
    if len(ray_lines) < 2 or N < 2: return None
    height, width = frame.shape[:2]
    num_to_sample = min(N, len(ray_lines))
    selected_lines = random.sample(ray_lines, num_to_sample)
    new_intersections_this_frame = []
    for i in range(len(selected_lines) - 1):
        line1, line2 = selected_lines[i], selected_lines[i + 1]
        if not isinstance(line1, (tuple, list)) or len(line1) != 3 or \
           not isinstance(line2, (tuple, list)) or len(line2) != 3: continue
        try: angle1, angle2 = line1[2], line2[2]
        except IndexError: continue
        if abs(angle1 - angle2) >= 2.0:
            intersection = find_line_intersection(line1, line2)
            if intersection:
                ix, iy = intersection
                if (0 <= ix < width) and (0 <= iy < height):
                    dist = math.hypot(ix - current_center_avg[0], iy - current_center_avg[1])
                    if dist < MAX_INTERSECTION_DISTANCE:
                        new_intersections_this_frame.append(intersection)
                        stored_intersections.append(intersection)
    if len(stored_intersections) > M:
        compute_average_intersection.stored_intersections = stored_intersections[-M:]
    current_history = compute_average_intersection.stored_intersections
    if not current_history: return None
    avg_x = np.mean([pt[0] for pt in current_history])
    avg_y = np.mean([pt[0] for pt in current_history])
    return (int(avg_x), int(avg_y))


def find_line_intersection(ellipse1, ellipse2):
    try:
        (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
        (cx2, cy2), (_, minor_axis2), angle2 = ellipse2
        if minor_axis1 <= 0 or minor_axis2 <= 0: return None
        angle1_rad, angle2_rad = np.deg2rad(angle1), np.deg2rad(angle2)
        dx1 = (minor_axis1 / 2.0) * np.cos(angle1_rad); dy1 = (minor_axis1 / 2.0) * np.sin(angle1_rad)
        dx2 = (minor_axis2 / 2.0) * np.cos(angle2_rad); dy2 = (minor_axis2 / 2.0) * np.sin(angle2_rad)
        A = np.array([[dx1, -dx2], [dy1, -dy2]]); B = np.array([cx2 - cx1, cy2 - cy1])
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-6: return None
        solution = np.linalg.solve(A, B); t1 = solution[0]
        intersection_x = cx1 + t1 * dx1; intersection_y = cy1 + t1 * dy1
        return (int(round(intersection_x)), int(round(intersection_y)))
    except (ValueError, TypeError, np.linalg.LinAlgError, IndexError): return None

def process_frame(frame):
    """Función envoltorio simplificada"""
    global clahe 
    frame = crop_to_aspect_ratio(frame)
    gray_frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_blurred = cv2.GaussianBlur(gray_frame_original, GAUSSIAN_KERNEL_SIZE, 0)
    gray_frame_clahe = clahe.apply(gray_frame_blurred)
    
    # --- ¡MODIFICADO! Recibir 4 valores ---
    processed_frame, model_center, pupil_center, scaled_radius = process_frames(frame, gray_frame_clahe)
    
    return processed_frame, model_center, pupil_center, scaled_radius


# =================================================================
# THREAD DE VISUALIZACIÓN 3D (Sin cambios)
# =================================================================
def run_gl_thread():
    """Inicia la aplicación PyQt/OpenGL en un hilo separado."""
    global app
    app = start_gl_window()
    gl_ready.set()
    sys.exit(app.exec_())

# =================================================================
# CÓDIGO DE PROCESAMIENTO (Modificado)
# =================================================================
def process_frame_for_comparison(frame):
    """
    Procesa un frame usando la lógica robusta DE utils y devuelve datos 
    para visualización y el frame procesado.
    """
    frame_shape_after_crop = crop_to_aspect_ratio(frame).shape
    
    # --- ¡MODIFICADO! Recibir 4 valores ---
    processed_frame, model_center, pupil_center, scaled_radius = process_frame(frame) 
    
    # --- ¡MODIFICADO! Devolver 5 valores ---
    return processed_frame, model_center, pupil_center, frame_shape_after_crop, scaled_radius


# =================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO DE VIDEO (Modificada)
# =================================================================
def process_video_comparison(video_path, output_video_path):
    
    # --- Resetear las variables de estado ---
    global ray_lines, model_centers, stable_pupil_centers, prev_model_center_avg
    global last_known_pupil_center, frames_since_last_good_detection 
    
    ray_lines, model_centers, stable_pupil_centers = [], [], []
    prev_model_center_avg = (320, 240)
    last_known_pupil_center = None
    frames_since_last_good_detection = 0
    
    if hasattr(compute_average_intersection, 'stored_intersections'):
        compute_average_intersection.stored_intersections = []
    # --- Fin del reseteo ---
    
    if not os.path.exists(video_path):
        print(f"Error: Video '{video_path}' no existe.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video '{video_path}'.")
        return
    
    # --- Configuración del VideoWriter (Sin cambios) ---
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30.0
    output_width = 640 + 640; output_height = 480
    output_dims = (output_width, output_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, output_dims)
    if not video_writer.isOpened():
        print(f"Error: No se pudo abrir el VideoWriter para {output_video_path}")
        cap.release()
        return
    # --- Fin de la configuración ---
    
    gl_thread = threading.Thread(target=run_gl_thread)
    gl_thread.start()
    
    print("⏳ Esperando que la ventana 3D de OpenGL se inicie...")
    gl_ready.wait()
    print("✅ Ventana 3D lista.")

    print(f"Procesando: {video_path}")
    print(f"💾 Guardando video combinado en: {output_video_path}")
    print("\nControles:\n   'q' - Salir\n   'ESPACIO' - Pausar/Reanudar\n" + "-"*60)
    
    frame_number = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n⏹️   Fin del video alcanzado")
                    break
                
                try:
                    # 1. Procesar frame y obtener los datos
                    # --- ¡MODIFICADO! Recibir 5 valores ---
                    processed_frame, model_center, pupil_center, frame_shape, scaled_radius = \
                        process_frame_for_comparison(frame)
                    
                    # 2. ENVIAR los datos al hilo de OpenGL
                    # --- ¡MODIFICADO! Enviar siempre datos al 3D ---
                    if model_center:
                        pupil_x_to_send = pupil_center[0] if pupil_center else -1
                        pupil_y_to_send = pupil_center[1] if pupil_center else -1

                        update_sphere_rotation_signal(
                            pupil_x_to_send, pupil_y_to_send, 
                            model_center[0], model_center[1], 
                            frame_shape[1], frame_shape[0],
                            scaled_radius # <-- ¡AÑADIDO!
                        )
                    
                    # 3. RECUPERAR la última imagen renderizada de OpenGL (Sin cambios)
                    gl_image = get_latest_rendered_image()
                    
                    h_cv, w_cv, _ = processed_frame.shape # (480, 640, 3)

                    # 4. PREPARAR VISTA 3D (Sin cambios)
                    if gl_image is not None:
                        clean_view = cv2.cvtColor(gl_image, cv2.COLOR_RGB2BGR)
                        if clean_view.shape[0] != h_cv or clean_view.shape[1] != w_cv:
                            clean_view = cv2.resize(clean_view, (w_cv, h_cv), interpolation=cv2.INTER_LINEAR)
                    else:
                        clean_view = np.zeros((h_cv, w_cv, 3), dtype=np.uint8) 

                    # 5. COMBINAR, MOSTRAR Y GUARDAR (Sin cambios)
                    combined_frame = np.hstack((processed_frame, clean_view))
                    cv2.imshow("Eye Tracking: Real vs. 3D Model", combined_frame)
                    video_writer.write(combined_frame)
                    
                    frame_number += 1
                    if frame_number % 30 == 0:
                        print(f"📊 Frame {frame_number} procesado", end='\r')
                            
                except Exception as e:
                    print(f"\n⚠️   Error procesando frame {frame_number}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                print("\n🛑 Detenido por usuario")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"\n{'⏸️   PAUSADO' if paused else '▶️   REPRODUCIENDO'}")
    
    finally:
        cap.release()
        if 'video_writer' in locals() and video_writer.isOpened():
            video_writer.release()
            print(f"\n✅ Video guardado exitosamente en {output_video_path}")
            
        cv2.destroyAllWindows()
        if app is not None:
            app.quit()
        if gl_thread.is_alive():
             gl_thread.join(timeout=1)
        print("\n✅ Procesamiento completado")
        print(f"📊 Total frames: {frame_number}")

def main():
    print("="*60 + "\nCOMPARADOR DE EYE TRACKING - OJO REAL VS VIRTUAL (MODO DEMO)\n" + "="*60)
    print("🚀 Usando lógica de detección de 'eye_tracker_utils.py' (Mejor Fit Elíptico)")
    
    video_path = ""
    output_path = ""

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if len(sys.argv) > 2:
             output_path = sys.argv[2]
        else:
             base, _ = os.path.splitext(os.path.basename(video_path))
             output_path = f"{base}_combinado.mp4"
             print(f"ℹ️   No se especificó ruta de salida. Usando: {output_path}")
    else:
        video_path = r"C:\Users\Victor\Documents\Tesis3D\Videos\Experimento_1\Victor\Victor3_intento_1.mp4"
        output_path = "video_combinado_default.mp4"
        print(f"ℹ️   Usando rutas por defecto. Video: {video_path}")
    
    process_video_comparison(video_path, output_path)

if __name__ == "__main__":
    main()