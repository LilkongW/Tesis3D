import cv2
import random
import math
import numpy as np
import os
import time
import csv

# --- PARÁMETROS DE FILTRADO Y PREPROCESAMIENTO ---
FIXED_THRESHOLD_VALUE = 50    # Umbral fijo (BAJADO SEGÚN TU SCRIPT DE DEBUG)
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0         # Límite de clip
MORPH_KERNEL_SIZE = 5          # Tamaño kernel morfología
# ------------------------------------------
# --- FILTROS DE ÁREA, FIT, BBOX, Y FALLBACK ELIMINADOS ---
# ------------------------------------------

# --- PARÁMETRO DE ESTABILIDAD DEL MODELO ---
MAX_INTERSECTION_DISTANCE = 30 
# ------------------------------------------

# --- PARÁMETROS DE ESTABILIDAD TEMPORAL (FILTRO DE PARPADEO) ---
MAX_PUPIL_JUMP_DISTANCE = 120 # (Píxeles) Distancia máxima
MAX_LOST_TRACK_FRAMES = 6    # (Frames) N. de frames para resetear
# -------------------------------------------------------------------------

# --- <<<--- PARÁMETROS DE BÚSQUEDA ELÍPTICA (Tus valores) ---
IRIS_RADIAL_RAYS = 450          
POSITIVE_GRADIENT_THRESHOLD = 2   
RING_END_THRESHOLD = -2           
PUPIL_SCALE_START = 1.0           
PUPIL_SCALE_END = 6            
NUM_ELLIPTICAL_STEPS = 100        
# --- <<<--- ---

# --- ¡NUEVOS! PARÁMETROS DE FILTRADO (Basado en Radio Normalizado) ---
MIN_NORMALIZED_RADIUS = 1.5    # (Filtro 1a) Mínimo 1.2x el tamaño de la pupila
MAX_NORMALIZED_RADIUS = 5    # (Filtro 1b) ¡NUEVO! Máximo 4.5x (antes de 5.0)
ROBUST_FILTER_THRESHOLD = 2    # (Filtro 2) Umbral de pertenencia (Z-score)
MORPH_CLEANUP_KERNEL_SIZE = 7    # (Filtro 3) Kernel morfológico
# --- <<<--- ---

# --- ETAPA 5 (SUAVIZADO) ---
IRIS_SMOOTHING_ALPHA = 0.33 # Alpha para EMA (Equivalente a 5 frames: 2 / (5 + 1))
# --- <<<--- ---


# --- VARIABLES DE ESTADO GLOBALES ---
ray_lines = []
model_centers = []
stable_pupil_centers = []
max_rays = 120
prev = (280, 150)
max_observed_distance = 240 

# --- VARIABLES GLOBALES DE ESTABILIDAD ---
last_known_pupil_center = None
frames_since_last_good_detection = 0
# --- <<<--- NUEVA VARIABLE DE ESTADO GLOBAL DEL IRIS ---
smoothed_iris_ellipse = ((0,0),(0,0),0) # Para el suavizado EMA
# --- <<<--- ---

# --- FUNCIONES DE PROCESAMIENTO ---

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

# --- FUNCIÓN DE OPTIMIZACIÓN DE ÁNGULO (CORREGIDA) ---
def optimize_contours_by_angle(contours):
    if not isinstance(contours, list) or len(contours) < 1 or len(contours[0]) < 5: return np.array([], dtype=np.int32).reshape((-1, 1, 2))
    
    # --- ¡LÍNEA CORREGIDA! ---
    if len(contours[0].shape) == 2: 
        all_contours = contours[0].reshape((-1, 1, 2))
    # --- FIN DE LA CORRECCIÓN ---
    else: 
        all_contours = contours[0]
    
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


# --- FUNCIÓN DE FALLBACK (ELIMINADA) ---
# ... (find_fallback_pupil eliminada) ...


# --- FUNCIÓN process_frames (LÓGICA DE DETECCIÓN REESCRITA) ---
def process_frames(frame, gray_frame_clahe):
    global ray_lines, max_rays, prev_model_center_avg, max_observed_distance, stable_pupil_centers, model_centers
    global last_known_pupil_center, frames_since_last_good_detection
    global smoothed_iris_ellipse 

    # --- data_dict actualizado ---
    data_dict = {
        "valid_deteccion": False, "sphere_center_x": None, "sphere_center_y": None, "sphere_center_z": None,
        "pupil_center_x": None, "pupil_center_y": None,
        "gaze_x": None, "gaze_y": None, "gaze_z": None,
        "ellipse_width": None, "ellipse_height": None, "ellipse_angle": None,
        "contour_area": None
    }
    
    h_frame, w_frame = frame.shape[:2] # Para los límites de los rayos

    # --- INICIO: NUEVA LÓGICA DE DETECCIÓN (MÉTODO ÚNICO) ---
    
    # 1. Encontrar el centro del bloque 5x5 más oscuro ("Semilla")
    avg_intensity_map = cv2.blur(gray_frame_clahe, (5, 5))
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(avg_intensity_map)
    darkest_center_point = minLoc

    # 2. Binarización y Morfología
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    thresholded_image_raw = apply_fixed_binary_threshold(gray_frame_clahe, FIXED_THRESHOLD_VALUE)
    thresholded_image_closed = cv2.morphologyEx(thresholded_image_raw, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    thresholded_image_final = cv2.morphologyEx(thresholded_image_closed, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # 3. Encontrar Contornos
    contours, _ = cv2.findContours(thresholded_image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Encontrar el contorno que contenga la "Semilla"
    best_pupil_contour = None
    best_contour_area = 0.0

    if darkest_center_point:
        for contour in contours:
            # Comprobar si la "semilla" (punto más oscuro) está DENTRO de este contorno
            if cv2.pointPolygonTest(contour, darkest_center_point, False) >= 0:
                best_pupil_contour = contour
                best_contour_area = cv2.contourArea(best_pupil_contour)
                break # Encontramos nuestro contorno, salimos del bucle
    
    # --- FIN: NUEVA LÓGICA DE DETECCIÓN ---

    # 5. Procesar si se encontró un contorno válido
    final_rotated_rect = None # Esta es nuestra elipse de pupila
    center_x, center_y = None, None
    is_detection_temporally_stable = False 

    if best_pupil_contour is not None: 
        data_dict["contour_area"] = best_contour_area 
        
        optimized_contour = optimize_contours_by_angle([best_pupil_contour])
        ellipse = None
        try:
            if len(optimized_contour) >= 5:
                ellipse = cv2.fitEllipse(optimized_contour)
            elif len(best_pupil_contour) >= 5:
                ellipse = cv2.fitEllipse(best_pupil_contour)
        except cv2.error: ellipse = None

        if ellipse is not None:
            final_rotated_rect = ellipse
            center_x_raw, center_y_raw = map(int, final_rotated_rect[0])
            stable_pupil_center = update_and_average_point(stable_pupil_centers, (center_x_raw, center_y_raw), N=2)
            center_x, center_y = stable_pupil_center if stable_pupil_center else (center_x_raw, center_y_raw)
            
            # --- <<<--- INICIO: LÓGICA DE DETECCIÓN DE IRIS (SIN CAMBIOS) ---
            
            iris_edge_points = []
            cleaned_points = [] 
            current_fitted_ellipse = None
            
            (cx_f, cy_f), (w_axis, h_axis), angle_deg = final_rotated_rect
            cx, cy = int(cx_f), int(cy_f)
            
            semi_a = w_axis / 2.0
            semi_b = h_axis / 2.0
            if semi_a == 0: semi_a = 1.0
            if semi_b == 0: semi_b = 1.0
            
            angle_rad = np.deg2rad(angle_deg)
            cos_rot = np.cos(angle_rad)
            sin_rot = np.sin(angle_rad)
            
            # --- 3. Lanzar "Rayos" Elípticos ---
            for i in range(IRIS_RADIAL_RAYS):
                t = (i / IRIS_RADIAL_RAYS) * 2 * np.pi
                cos_t = np.cos(t)
                sin_t = np.sin(t)

                s_start = PUPIL_SCALE_START # s = 1.0
                x_local_start = (semi_a * s_start) * cos_t
                y_local_start = (semi_b * s_start) * sin_t
                x_rot_start = x_local_start * cos_rot - y_local_start * sin_rot
                y_rot_start = x_local_start * sin_rot + y_local_start * cos_rot
                
                x_prev = int(cx + x_rot_start)
                y_prev = int(cy + y_rot_start)
                
                if not (0 <= x_prev < w_frame and 0 <= y_prev < h_frame):
                    continue
                prev_intensity = int(gray_frame_clahe[y_prev, x_prev])
                
                state = "SEARCHING"

                # --- 4. Caminar a lo largo del "rayo" elíptico ---
                for step in range(1, NUM_ELLIPTICAL_STEPS + 1):
                    
                    s = PUPIL_SCALE_START + (step / NUM_ELLIPTICAL_STEPS) * (PUPIL_SCALE_END - PUPIL_SCALE_START)
                    
                    x_local = (semi_a * s) * cos_t
                    y_local = (semi_b * s) * sin_t
                    x_rotated = x_local * cos_rot - y_local * sin_rot
                    y_rotated = x_local * sin_rot + y_local * cos_rot
                    
                    x_curr = int(cx + x_rotated)
                    y_curr = int(cy + y_rotated)
                    
                    if not (0 <= x_curr < w_frame and 0 <= y_curr < h_frame):
                        break 
                    
                    if x_curr == x_prev and y_curr == y_prev:
                        continue
                        
                    curr_intensity = int(gray_frame_clahe[y_curr, x_curr])
                    gradient = curr_intensity - prev_intensity
                    
                    if state == "SEARCHING":
                        if gradient > POSITIVE_GRADIENT_THRESHOLD:
                            state = "IGNORING_FIRST_RING"
                    
                    elif state == "IGNORING_FIRST_RING":
                        if gradient < RING_END_THRESHOLD:
                            state = "READY_TO_PAINT"
                    
                    elif state == "READY_TO_PAINT":
                        if gradient > POSITIVE_GRADIENT_THRESHOLD:
                            iris_edge_points.append((x_curr, y_curr))
                            state = "DONE"

                    elif state == "DONE":
                        break
                        
                    prev_intensity = curr_intensity
                    x_prev, y_prev = x_curr, y_curr

            # --- 5. FILTRO DE TRES ETAPAS ---
            if len(iris_edge_points) > 0:
                
                # --- Pre-cálculo: Calcular Radio Normalizado ---
                points_data = [] 
                for pt in iris_edge_points:
                    x_p, y_p = pt
                    x_trans = x_p - cx
                    y_trans = y_p - cy
                    x_derot = x_trans * cos_rot + y_trans * sin_rot
                    y_derot = -x_trans * sin_rot + y_trans * cos_rot
                    x_norm = x_derot / semi_a
                    y_norm = y_derot / semi_b
                    norm_radius = np.sqrt(x_norm**2 + y_norm**2)
                    points_data.append((pt, norm_radius))

                # --- ETAPA 1: Filtro de Banda de Paso (Estático) ---
                stage_one_inliers_pts = [] 
                stage_one_inliers_rad = [] 
                
                for pt, r_norm in points_data:
                    if MIN_NORMALIZED_RADIUS <= r_norm <= MAX_NORMALIZED_RADIUS:
                        stage_one_inliers_pts.append(pt)
                        stage_one_inliers_rad.append(r_norm)
                
                # --- ETAPA 2: Filtro Dinámico (Mediana) ---
                stage_two_inliers_pts = [] 

                if len(stage_one_inliers_pts) >= 5: 
                    median_radius = np.median(stage_one_inliers_rad)
                    mad = np.median(np.abs(stage_one_inliers_rad - median_radius)) * 1.4826 
                    if mad == 0: mad = 1.0 
                    
                    for i, pt in enumerate(stage_one_inliers_pts):
                        r_norm = stage_one_inliers_rad[i]
                        z_score_robusta = np.abs(r_norm - median_radius) / mad
                        
                        if z_score_robusta < ROBUST_FILTER_THRESHOLD:
                            stage_two_inliers_pts.append(pt)
                else:
                    stage_two_inliers_pts = stage_one_inliers_pts

                # --- ETAPA 3: Limpieza Morfológica ---
                if len(stage_two_inliers_pts) > 0:
                    point_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
                    for pt in stage_two_inliers_pts:
                        cv2.circle(point_mask, pt, 1, 255, -1) 
                        
                    kernel_size = MORPH_CLEANUP_KERNEL_SIZE
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    cleaned_mask = cv2.morphologyEx(point_mask, cv2.MORPH_CLOSE, kernel)
                    
                    coords = np.where(cleaned_mask == 255)
                    cleaned_points = list(zip(coords[1], coords[0])) 

                # --- ETAPA 4: AJUSTE DE ELIPSE ---
                if len(cleaned_points) >= 5:
                    try:
                        final_points_np = np.array(cleaned_points, dtype=np.int32).reshape((-1, 1, 2))
                        current_fitted_ellipse = cv2.fitEllipse(final_points_np) 
                    except cv2.error as e:
                        current_fitted_ellipse = None
                
                # --- ETAPA 5: SUAVIZADO (EMA) Y FORZADO CONCÉNTRICO ---
                if current_fitted_ellipse is not None:
                    pupil_center = final_rotated_rect[0] 
                    if smoothed_iris_ellipse[1][0] == 0.0: 
                        smoothed_iris_ellipse = (pupil_center, current_fitted_ellipse[1], current_fitted_ellipse[2])
                    else:
                        alpha = IRIS_SMOOTHING_ALPHA
                        scx, scy = pupil_center
                        sax = (current_fitted_ellipse[1][0] * alpha) + (smoothed_iris_ellipse[1][0] * (1.0 - alpha))
                        say = (current_fitted_ellipse[1][1] * alpha) + (smoothed_iris_ellipse[1][1] * (1.0 - alpha))
                        sang = (current_fitted_ellipse[2] * alpha) + (smoothed_iris_ellipse[2] * (1.0 - alpha))
                        smoothed_iris_ellipse = ((scx, scy), (sax, say), sang)
            
            # --- <<<--- FIN: LÓGICA DE DETECCIÓN DE IRIS ---
            
            # --- FILTRO 3: TEMPORAL (PARPADEO/SALTO) ---
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
        # Esto se ejecuta si el método de semilla NO encontró un contorno
        frames_since_last_good_detection += 1
        smoothed_iris_ellipse = ((0,0),(0,0),0) # Resetear el iris si se pierde la pupila

    # Calcular centro del modelo (Esfera Cian)
    model_center_average = prev_model_center_avg
    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5, model_center_average)
    if model_center is not None:
        model_center_average = update_and_average_point(model_centers, model_center, 800)
        prev_model_center_avg = model_center_average
    data_dict["sphere_center_x"] = model_center_average[0]
    data_dict["sphere_center_y"] = model_center_average[1]

    # --- Dibujar y poblar datos SÓLO si la detección es VÁLIDA EN TODOS LOS FILTROS ---
    if is_detection_temporally_stable: 
        
        # --- FILTRO 4: ESPACIAL (LÍMITE DEL MODELO) ---
        dist_from_sphere_center = math.hypot(center_x - model_center_average[0], 
                                             center_y - model_center_average[1])
        
        if dist_from_sphere_center <= max_observed_distance:
            
            # --- ¡DETECCIÓN FINAL VÁLIDA! ---
            ray_lines.append(final_rotated_rect) 
            if len(ray_lines) > max_rays: ray_lines.pop(0)

            # Dibujar Pupila
            cv2.ellipse(frame, final_rotated_rect, (0, 255, 255), 2) # Amarillo
            
            # Dibujar Gaze
            cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2) # Azul claro
            dx = center_x - model_center_average[0]; dy = center_y - model_center_average[1]
            ex = int(model_center_average[0] + 2 * dx); ey = int(model_center_average[1] + 2 * dy)
            cv2.line(frame, (center_x, center_y), (ex, ey), (200, 255, 0), 3) # Verde lima

            # Calcular Gaze 3D
            center_3d, direction_3d = compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1], max_observed_distance)

            if center_3d is not None and direction_3d is not None:
                # Poblar el diccionario de datos
                data_dict["valid_deteccion"] = True
                data_dict["sphere_center_z"] = center_3d[2]
                data_dict["pupil_center_x"] = center_x
                data_dict["pupil_center_y"] = center_y
                data_dict["gaze_x"] = direction_3d[0]; data_dict["gaze_y"] = direction_3d[1]; data_dict["gaze_z"] = direction_3d[2]
                data_dict["ellipse_width"] = final_rotated_rect[1][0]; data_dict["ellipse_height"] = final_rotated_rect[1][1]; data_dict["ellipse_angle"] = final_rotated_rect[2]
                
                origin_text = f"Origin: ({center_3d[0]:.2f}, {center_3d[1]:.2f}, {center_3d[2]:.2f})"
                dir_text = f"Direction: ({direction_3d[0]:.2f}, {direction_3d[1]:.2f}, {direction_3d[2]:.2f})"
                cv2.putText(frame, origin_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, dir_text, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # --- DIBUJAR SIEMPRE (FUERA DE LOS FILTROS) ---
    
    # Dibujar modelo del ojo
    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2) # Azul oscuro
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1) # Cian

    # --- <<<--- DIBUJAR IRIS SUAVIZADO (SIEMPRE) ---
    if smoothed_iris_ellipse[1][0] > 0 and smoothed_iris_ellipse[1][1] > 0:
        cv2.ellipse(frame, smoothed_iris_ellipse, (255, 0, 255), 2) # Magenta
    # --- <<<--- ---
    
    # Mostrar frame
    cv2.imshow("Frame with Ellipse and Rays", frame)
    return data_dict

# --- OTRAS FUNCIONES UTILITARIAS (SIN CAMBIOS) ---
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
    avg_y = np.mean([pt[1] for pt in current_history])
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

def compute_gaze_vector(x_pupil, y_pupil, x_sphere, y_sphere, max_radius_pixels, screen_width=640, screen_height=480):
    try:
        sphere_offset_x = (float(x_sphere) / screen_width) * 2.0 - 1.0
        sphere_offset_y = 1.0 - (float(y_sphere) / screen_height) * 2.0
        sphere_center_3d = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])
        dx = float(x_pupil) - float(x_sphere); dy = float(y_pupil) - float(y_sphere)
        if max_radius_pixels <= 0: max_radius_pixels = 1.0
        gaze_x = dx / max_radius_pixels; gaze_y = -dy / max_radius_pixels
        mag_sq_2d = gaze_x**2 + gaze_y**2
        if mag_sq_2d > 1.0:
            mag_2d = np.sqrt(mag_sq_2d); gaze_x /= mag_2d; gaze_y /= mag_2d; mag_sq_2d = 1.0
        gaze_z_sq = max(0.0, 1.0 - mag_sq_2d) 
        gaze_z = -np.sqrt(gaze_z_sq)
        gaze_direction_3d = np.array([gaze_x, gaze_y, gaze_z])
        norm = np.linalg.norm(gaze_direction_3d)
        if norm < 1e-6: return sphere_center_3d, np.array([0.0, 0.0, -1.0])
        gaze_direction_3d /= norm
        return sphere_center_3d, gaze_direction_3d
    except Exception as e:
         fallback_center = np.array([0.0, 0.0, 0.0]); fallback_gaze = np.array([0.0, 0.0, -1.0])
         return fallback_center, fallback_gaze

# --- FUNCIÓN DE PROCESAMIENTO PRINCIPAL (CORREGIDA) ---
def process_frame(frame):
    frame = crop_to_aspect_ratio(frame)
    # --- <<<--- ¡LÍNEA CORREGIDA! ---
    gray_frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # --- <<<--- ---
    gray_frame_blurred = cv2.GaussianBlur(gray_frame_original, GAUSSIAN_KERNEL_SIZE, 0)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    gray_frame_clahe = clahe.apply(gray_frame_blurred)
    data_dict = process_frames(frame, gray_frame_clahe)
    return data_dict

# --- FUNCIÓN DE PROCESAMIENTO DE VIDEO (MODIFICADA PARA CSV Y RESETEO) ---
def process_video_from_path(video_path, video_name, csv_path,prev):
    # --- MODIFICADO: Resetear todas las variables de estado ---
    global ray_lines, model_centers, stable_pupil_centers, prev_model_center_avg
    global last_known_pupil_center, frames_since_last_good_detection 
    global smoothed_iris_ellipse # <-- ¡NUEVO RESETEO!
    
    ray_lines, model_centers, stable_pupil_centers = [], [], []
    prev_model_center_avg = prev
    last_known_pupil_center = None
    frames_since_last_good_detection = 0
    smoothed_iris_ellipse = ((0,0),(0,0),0) # <-- ¡NUEVO RESETEO!
    # ---------------------

    if hasattr(compute_average_intersection, 'stored_intersections'):
        compute_average_intersection.stored_intersections = []
    min_area_found = float('inf'); max_area_found = float('-inf')

    if not os.path.exists(video_path): print(f"Error: Video file not found at {video_path}"); return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video file {video_path}"); return

    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30.0
    frame_delay = int(1000 / fps); frame_counter = 0

    print(f"Processing video: {video_path}\nSaving CSV data to: {csv_path}\nPress 'q' to quit, 'space' to pause")

    # --- <<<--- ENCABEZADO CSV ACTUALIZADO ---
    csv_header = [
        "video_name", "frame_number", "timestamp_ms", "valid_deteccion",
        "sphere_center_x", "sphere_center_y", "sphere_center_z",
        "pupil_center_x", "pupil_center_y",
        "gaze_x", "gaze_y", "gaze_z",
        "ellipse_width", "ellipse_height", "ellipse_angle",
        "contour_area"
        # (Columnas de Iris 'mm' eliminadas)
    ]
    # --- <<<--- ---

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret: break
                frame_counter += 1
                timestamp_ms = (frame_counter / fps) * 100.0
                data = process_frame(frame)
                
                # --- Lógica de seguimiento de área (AHORA OCURRE DESPUÉS) ---
                if data.get("valid_deteccion") and data.get("contour_area") is not None:
                    current_area = data["contour_area"]
                    min_area_found = min(min_area_found, current_area)
                    max_area_found = max(max_area_found, current_area)
                
                # --- <<<--- FILA DE CSV ACTUALIZADA ---
                row = [
                    video_name, frame_counter, f"{timestamp_ms:.3f}", data.get("valid_deteccion", False),
                    f"{data.get('sphere_center_x', ''):.3f}" if data.get('sphere_center_x') is not None else '',
                    f"{data.get('sphere_center_y', ''):.3f}" if data.get('sphere_center_y') is not None else '',
                    f"{data.get('sphere_center_z', ''):.3f}" if data.get('sphere_center_z') is not None else '',
                    f"{data.get('pupil_center_x', ''):.3f}" if data.get('pupil_center_x') is not None else '',
                    f"{data.get('pupil_center_y', ''):.3f}" if data.get('pupil_center_y') is not None else '',
                    f"{data.get('gaze_x', ''):.6f}" if data.get('gaze_x') is not None else '', 
                    f"{data.get('gaze_y', ''):.6f}" if data.get('gaze_y') is not None else '',
                    f"{data.get('gaze_z', ''):.6f}" if data.get('gaze_z') is not None else '',
                    f"{data.get('ellipse_width', ''):.3f}" if data.get('ellipse_width') is not None else '',
                    f"{data.get('ellipse_height', ''):.3f}" if data.get('ellipse_height') is not None else '',
                    f"{data.get('ellipse_angle', ''):.3f}" if data.get('ellipse_angle') is not None else '',
                    f"{data.get('contour_area', ''):.1f}" if data.get('contour_area') is not None else ''
                    # (Columnas de Iris 'mm' eliminadas)
                ]
                # --- <<<--- ---
                
                writer.writerow(row)
                processing_duration_ms = (time.time() - start_time) * 1000
                wait_time = max(1, frame_delay - int(processing_duration_ms))
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'): print("Processing stopped by user."); break
                elif key == ord(' '): print("Paused. Press any key to continue..."); cv2.waitKey(0)
    except IOError as e: print(f"Error writing to CSV file {csv_path}: {e}")
    except Exception as e: print(f"An unexpected error occurred during processing: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        print(f"--- [COMPLETED] Video: {video_name} ---")
        if max_area_found == float('-inf'): print("     -> No valid pupils detected in this video.")
        else: print(f"     -> Min Contour Area: {min_area_found:.1f}, Max Contour Area: {max_area_found:.1f}")

# --- ENTRY POINT (Example - keep commented out for utils script) ---
# if __name__ == "__main__":
#     ...