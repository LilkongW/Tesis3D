import cv2
import random
import math
import numpy as np
import os
import time
import csv
from ultralytics import YOLO

# ==========================================
#      CONFIGURACIÓN DE CONTROL (USER)
# ==========================================
ENABLE_IRIS_PROCESSING = False   # True: Calcula iris. False: Solo pupila (más rápido).
SHOW_VISUALIZATION = True       # True: Muestra ventana. False: Modo rápido.

# Configuración de Limpieza de Parpadeos
BLINK_PADDING_FRAMES = 3         # Cuántos frames borrar antes y después de un parpadeo

# ==========================================

# --- PARÁMETROS DE FILTRADO Y PREPROCESAMIENTO ---
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0

# ------------------------------------------
# Definir la ruta base
if os.name == 'nt': 
    BASE_DIR = r"C:\Users\Victor\Documents\Tesis3D"
else: 
    BASE_DIR = r"/home/vit/Documentos/Tesis3D"

# --- RUTAS ---
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
RETRAIN_FRAMES_DIR = os.path.join(BASE_DIR, "frames_para_retrain")

# --- PARÁMETROS YOLO ---
YOLO_MIN_CONFIDENCE = 0.5  
YOLO_ROI_EXPANSION_PX = 5 

# --- UMBRALES ---
PUPIL_FIXED_THRESHOLD = 20
MAX_INTERSECTION_DISTANCE = 10
MAX_PUPIL_JUMP_DISTANCE = 120
MAX_LOST_TRACK_FRAMES = 6

# --- PARÁMETROS IRIS ---
IRIS_RADIAL_RAYS = 450
POSITIVE_GRADIENT_THRESHOLD = 2
RING_END_THRESHOLD = -2
PUPIL_SCALE_START = 1.0
PUPIL_SCALE_END = 6
NUM_ELLIPTICAL_STEPS = 100
MIN_NORMALIZED_RADIUS = 1.5
MAX_NORMALIZED_RADIUS = 5
ROBUST_FILTER_THRESHOLD = 2
MORPH_CLEANUP_KERNEL_SIZE = 7

# --- VARIABLES DE ESTADO GLOBALES ---
ray_lines = []
model_centers = []
max_rays = 120
prev = (160, 120)
max_observed_distance = 180
last_known_pupil_center = None
frames_since_last_good_detection = 0
smoothed_iris_ellipse = ((0, 0), (0, 0), 0)

# --- CARGA GLOBAL DEL MODELO ---
try:
    print(f"Cargando modelo YOLO desde: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    print("Modelo YOLO cargado exitosamente.")
    os.makedirs(RETRAIN_FRAMES_DIR, exist_ok=True)
except Exception as e:
    print(f"Error CRÍTICO al cargar el modelo YOLO: {e}")
    print("El script no podrá detectar la pupila.")
    model = None


# --- FUNCIONES DE PROCESAMIENTO ---

def crop_to_aspect_ratio(image, width=320, height=240):
    # Si la imagen ya tiene el tamaño exacto, la devolvemos inmediatamente.
    if image.shape[1] == width and image.shape[0] == height:
        return image

    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # La imagen es muy ancha (ej. 16:9): Recortar los lados
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        # La imagen es muy alta: Recortar arriba y abajo
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))


def apply_fixed_binary_threshold(image, threshold_value):
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

def optimize_contours_by_angle(contours):
    if not isinstance(contours, list) or len(contours) < 1 or len(contours[0]) < 5: return np.array([], dtype=np.int32).reshape((-1, 1, 2))
    if len(contours[0].shape) == 2:
        all_contours = contours[0].reshape((-1, 1, 2))
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


# --- FUNCIÓN process_frames ---
def process_frames(frame, gray_frame_clahe):
    global ray_lines, max_rays, prev_model_center_avg, max_observed_distance, model_centers
    global last_known_pupil_center, frames_since_last_good_detection
    global smoothed_iris_ellipse
    global model 

    data_dict = {
        "valid_deteccion": False, "sphere_center_x": None, "sphere_center_y": None, "sphere_center_z": None,
        "pupil_center_x": None, "pupil_center_y": None,
        "gaze_x": None, "gaze_y": None, "gaze_z": None,
        "ellipse_width": None, "ellipse_height": None, "ellipse_angle": None,
        "contour_area": None
    }
    
    h_frame, w_frame = frame.shape[:2]
    
    final_rotated_rect = None
    center_x, center_y = None, None
    is_detection_temporally_stable = False
    best_pupil_contour = None
    best_contour_area = 0.0
    expanded_bbox = (0, 0, w_frame, h_frame) 

    # --- 1. DETECCIÓN YOLO + ROI ---
    if model is None:
        print("Error: El modelo YOLO no está cargado. Saltando detección.")
    else:
        results = model.track(frame, persist=True, verbose=False)
        best_box = None
        max_conf = 0.0 

        if results[0].boxes:
            for box in results[0].boxes:
                if box.conf[0] > max_conf:
                    max_conf = box.conf[0]
                    best_box = box
        
        if best_box is not None: 
            x1_raw, y1_raw, x2_raw, y2_raw = best_box.xyxy[0].cpu().numpy().astype(int)
            x1_raw_exp = x1_raw - YOLO_ROI_EXPANSION_PX
            y1_raw_exp = y1_raw - YOLO_ROI_EXPANSION_PX
            x2_raw_exp = x2_raw + YOLO_ROI_EXPANSION_PX
            y2_raw_exp = y2_raw + YOLO_ROI_EXPANSION_PX
            x1, y1 = max(0, x1_raw_exp), max(0, y1_raw_exp)
            x2, y2 = min(w_frame, x2_raw_exp), min(h_frame, y2_raw_exp)
            expanded_bbox = (x1, y1, x2, y2) 
            
            if x1 < x2 and y1 < y2:
                roi = frame[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary_roi = cv2.threshold(gray_roi, PUPIL_FIXED_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
                contours_roi, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours_roi:
                    best_pupil_contour_roi = max(contours_roi, key=cv2.contourArea)
                    best_contour_area = cv2.contourArea(best_pupil_contour_roi)
                    best_pupil_contour = best_pupil_contour_roi + (x1, y1)

    # --- 2. ELIPSE Y PUPILA ---
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

        if final_rotated_rect is not None:
            center_x_raw, center_y_raw = map(int, final_rotated_rect[0])
            
            # Asignación directa (Raw Data)
            center_x, center_y = center_x_raw, center_y_raw
            
            # --- PROCESAMIENTO DE IRIS ---
            if ENABLE_IRIS_PROCESSING:
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
                for i in range(IRIS_RADIAL_RAYS):
                    t = (i / IRIS_RADIAL_RAYS) * 2 * np.pi
                    cos_t = np.cos(t)
                    sin_t = np.sin(t)
                    s_start = PUPIL_SCALE_START
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
                if len(iris_edge_points) > 0:
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
                    stage_one_inliers_pts = []
                    stage_one_inliers_rad = []
                    for pt, r_norm in points_data:
                        if MIN_NORMALIZED_RADIUS <= r_norm <= MAX_NORMALIZED_RADIUS:
                            stage_one_inliers_pts.append(pt)
                            stage_one_inliers_rad.append(r_norm)
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
                    if len(stage_two_inliers_pts) > 0:
                        point_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
                        for pt in stage_two_inliers_pts:
                            cv2.circle(point_mask, pt, 1, 255, -1)
                        kernel_size = MORPH_CLEANUP_KERNEL_SIZE
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                        cleaned_mask = cv2.morphologyEx(point_mask, cv2.MORPH_CLOSE, kernel)
                        coords = np.where(cleaned_mask == 255)
                        cleaned_points = list(zip(coords[1], coords[0]))
                    if len(cleaned_points) >= 5:
                        try:
                            final_points_np = np.array(cleaned_points, dtype=np.int32).reshape((-1, 1, 2))
                            current_fitted_ellipse = cv2.fitEllipse(final_points_np)
                        except cv2.error as e:
                            current_fitted_ellipse = None
                    if current_fitted_ellipse is not None:
                        pupil_center = final_rotated_rect[0]
                        smoothed_iris_ellipse = (pupil_center, current_fitted_ellipse[1], current_fitted_ellipse[2])

            # --- FILTRO 3: TEMPORAL (Solo para validación de saltos grandes) ---
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
        smoothed_iris_ellipse = ((0, 0), (0, 0), 0)

    # --- CÁLCULO DE GAZE ---
    model_center_average = prev_model_center_avg
    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5, model_center_average)
    if model_center is not None:
        model_center_average = update_and_average_point(model_centers, model_center, 800)
        prev_model_center_avg = model_center_average
    data_dict["sphere_center_x"] = model_center_average[0]
    data_dict["sphere_center_y"] = model_center_average[1]

    if is_detection_temporally_stable:
        dist_from_sphere_center = math.hypot(center_x - model_center_average[0],
                                             center_y - model_center_average[1])
        if dist_from_sphere_center <= max_observed_distance:
            ray_lines.append(final_rotated_rect)
            if len(ray_lines) > max_rays: ray_lines.pop(0)
            
            center_3d, direction_3d = compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1], max_observed_distance)
            if center_3d is not None and direction_3d is not None:
                data_dict["valid_deteccion"] = True
                data_dict["sphere_center_z"] = center_3d[2]
                data_dict["pupil_center_x"] = center_x
                data_dict["pupil_center_y"] = center_y
                data_dict["gaze_x"] = direction_3d[0]; data_dict["gaze_y"] = direction_3d[1]; data_dict["gaze_z"] = direction_3d[2]
                data_dict["ellipse_width"] = final_rotated_rect[1][0]; data_dict["ellipse_height"] = final_rotated_rect[1][1]; data_dict["ellipse_angle"] = final_rotated_rect[2]

    # --- VISUALIZACIÓN ---
    if SHOW_VISUALIZATION:
        if is_detection_temporally_stable and dist_from_sphere_center <= max_observed_distance:
            # Bbox y ROI
            (bx1, by1, bx2, by2) = expanded_bbox
            if best_box is not None:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 1)
            
            cv2.ellipse(frame, final_rotated_rect, (0, 255, 255), 2)
            cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)
            dx = center_x - model_center_average[0]; dy = center_y - model_center_average[1]
            ex = int(model_center_average[0] + 2 * dx); ey = int(model_center_average[1] + 2 * dy)
            cv2.line(frame, (center_x, center_y), (ex, ey), (200, 255, 0), 3)

            # Texto de Gaze
            if data_dict["valid_deteccion"]:
                origin_text = f"Origin: ({data_dict['sphere_center_x']:.2f}, {data_dict['sphere_center_y']:.2f}, {data_dict['sphere_center_z']:.2f})"
                dir_text = f"Direction: ({data_dict['gaze_x']:.2f}, {data_dict['gaze_y']:.2f}, {data_dict['gaze_z']:.2f})"
                cv2.putText(frame, origin_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, dir_text, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)
        cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)
        
        if smoothed_iris_ellipse[1][0] > 0 and smoothed_iris_ellipse[1][1] > 0:
            cv2.ellipse(frame, smoothed_iris_ellipse, (255, 0, 255), 2)
        
        # Estado detección
        if data_dict["valid_deteccion"]:
            status_text = "PUPILA DETECTADA"
            status_color = (0, 255, 0)
        else:
            status_text = "BUSCANDO..."
            status_color = (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow("Frame with Ellipse and Rays", frame)
    
    return data_dict

# --- OTRAS FUNCIONES UTILITARIAS ---
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

def process_frame(frame_recortado):
    gray_frame_original = cv2.cvtColor(frame_recortado, cv2.COLOR_BGR2GRAY)
    gray_frame_blurred = cv2.GaussianBlur(gray_frame_original, GAUSSIAN_KERNEL_SIZE, 0)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    gray_frame_clahe = clahe.apply(gray_frame_blurred)
    
    data_dict = process_frames(frame_recortado, gray_frame_clahe) 
    return data_dict

# --- NUEVA FUNCIÓN: LIMPIEZA DE PARPADEOS ---
def apply_blink_cleaning(rows, padding=3):
    """
    Identifica frames donde no hubo detección y marca N frames antes y después
    como inválidos para eliminar artefactos del párpado.
    """
    n = len(rows)
    # El índice 3 corresponde a 'valid_deteccion' en la lista 'row'
    invalid_indices = {i for i, row in enumerate(rows) if not row[3]}
    
    if not invalid_indices:
        return rows, 0
    
    indices_to_invalidate = set()
    for idx in invalid_indices:
        start = max(0, idx - padding)
        end = min(n, idx + padding + 1)
        for i in range(start, end):
            indices_to_invalidate.add(i)
            
    count = 0
    for idx in indices_to_invalidate:
        if rows[idx][3]: # Si era True, lo marcamos como False
            rows[idx][3] = False
            count += 1
            
    return rows, count

# --- FUNCIÓN DE PROCESAMIENTO DE VIDEO MODIFICADA ---
def process_video_from_path(video_path, video_name, csv_path, prev):
    # Variables globales
    global ray_lines, model_centers, prev_model_center_avg
    global last_known_pupil_center, frames_since_last_good_detection
    global smoothed_iris_ellipse 
    # Eliminado 'stable_pupil_centers' para evitar errores

    ray_lines, model_centers = [], []
    
    prev_model_center_avg = prev
    last_known_pupil_center = None
    frames_since_last_good_detection = 0
    smoothed_iris_ellipse = ((0, 0), (0, 0), 0)
    
    if hasattr(compute_average_intersection, 'stored_intersections'):
        compute_average_intersection.stored_intersections = []
    
    min_area_found = float('inf'); max_area_found = float('-inf')
    
    if not os.path.exists(video_path): print(f"Error: Video file not found at {video_path}"); return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video file {video_path}"); return
    
    # --- CORRECCIÓN DE FPS (CLAVE 1) ---
    REAL_FPS = 120.0 
    fps = REAL_FPS 
    
    frame_delay = int(1000 / fps); frame_counter = 0
    
    print(f"Processing video: {video_path}")
    print(f"Saving CSV data to: {csv_path}")
    if SHOW_VISUALIZATION:
        print("Visualization: ON (Press 'q' to quit, 'space' to pause)")
    else:
        print("Visualization: OFF (Processing at max speed)")

    csv_header = [
        "video_name", "frame_number", "timestamp_ms", "valid_deteccion",
        "sphere_center_x", "sphere_center_y", "sphere_center_z",
        "pupil_center_x", "pupil_center_y",
        "gaze_x", "gaze_y", "gaze_z",
        "ellipse_width", "ellipse_height", "ellipse_angle",
        "contour_area",
    ]
    
    # BUFFER DE MEMORIA PARA LIMPIEZA POST-PROCESO
    all_csv_rows = []

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: break
            frame_counter += 1
            
            # --- CORRECCIÓN DE TIMESTAMP (CLAVE 2) ---
            timestamp_ms = (frame_counter / fps) * 1000.0
            
            # --- CORRECCIÓN DE IMAGEN (CLAVE 3) ---
            frame_recortado_limpio = crop_to_aspect_ratio(frame)
            
            # Procesamos el frame
            data = process_frame(frame_recortado_limpio.copy())
            
            if not data.get("valid_deteccion", False):
                frame_filename = f"{video_name}_ts_{timestamp_ms:.0f}.jpg"
                save_path = os.path.join(RETRAIN_FRAMES_DIR, frame_filename)
                try:
                    cv2.imwrite(save_path, frame_recortado_limpio) 
                except Exception as e:
                    pass 

            if data.get("valid_deteccion") and data.get("contour_area") is not None:
                current_area = data["contour_area"]
                min_area_found = min(min_area_found, current_area)
                max_area_found = max(max_area_found, current_area)
                
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
                f"{data.get('contour_area', ''):.1f}" if data.get('contour_area') is not None else '',
            ]
            
            # EN LUGAR DE ESCRIBIR, GUARDAMOS EN RAM
            all_csv_rows.append(row)
            
            # --- CONTROL DE VELOCIDAD/VISUALIZACIÓN ---
            if SHOW_VISUALIZATION:
                processing_duration_ms = (time.time() - start_time) * 1000
                wait_time = max(1, frame_delay - int(processing_duration_ms))
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'): print("Processing stopped by user."); break
                elif key == ord(' '): print("Paused. Press any key to continue..."); cv2.waitKey(0)
            else:
                if frame_counter % 100 == 0:
                    print(f"Frame {frame_counter} procesado...")

        # --- FASE DE LIMPIEZA Y ESCRITURA ---
        print("   -> Aplicando limpieza de parpadeos...")
        clean_rows, cleaned_count = apply_blink_cleaning(all_csv_rows, padding=BLINK_PADDING_FRAMES)
        print(f"   -> {cleaned_count} frames de artefactos invalidados.")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(clean_rows)

    except IOError as e: print(f"Error writing to CSV file {csv_path}: {e}")
    except Exception as e: print(f"An unexpected error occurred during processing: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        print(f"--- [COMPLETED] Video: {video_name} ---")
        if max_area_found == float('-inf'): print("     -> No valid pupils detected in this video.")
        else: print(f"     -> Min Contour Area: {min_area_found:.1f}, Max Contour Area: {max_area_found:.1f}")