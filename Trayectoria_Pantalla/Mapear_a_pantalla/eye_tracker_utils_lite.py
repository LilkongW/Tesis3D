"""
Eye Tracker Utils Lite
Versión optimizada para detección de pupila y cálculo de vector de mirada.
Sin detección de iris ni funcionalidades de guardado de video.
"""

import cv2
import random
import math
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# PARÁMETROS DE CONFIGURACIÓN
# ==========================================

# Preprocesamiento
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0

# Detección YOLO
if os.name == 'nt':  # Windows
    BASE_DIR = r"C:\Users\Victor\Documents\Tesis3D"
else:  # Linux/Mac
    BASE_DIR = r"/home/vit/Documentos/Tesis3D"

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
YOLO_ROI_EXPANSION_PX = 5

# Binarización de pupila
PUPIL_FIXED_THRESHOLD = 14

# Estabilidad espacial
MAX_INTERSECTION_DISTANCE = 40

# Estabilidad temporal (anti-parpadeo)
MAX_PUPIL_JUMP_DISTANCE = 120
MAX_LOST_TRACK_FRAMES = 6

# ==========================================
# VARIABLES GLOBALES DE ESTADO
# ==========================================

ray_lines = []
model_centers = []
stable_pupil_centers = []
max_rays = 120
prev_model_center_avg = (280, 150)
max_observed_distance = 240
last_known_pupil_center = None
frames_since_last_good_detection = 0

# ==========================================
# CARGA DEL MODELO YOLO
# ==========================================

model = None
try:
    print(f"Cargando modelo YOLO desde: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    print("✓ Modelo YOLO cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar modelo YOLO: {e}")
    print("El sistema no podrá detectar la pupila")

# ==========================================
# FUNCIONES DE PREPROCESAMIENTO
# ==========================================

def crop_to_aspect_ratio(image, width=640, height=480):
    """Recorta y redimensiona imagen a proporción específica."""
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

def optimize_contours_by_angle(contours):
    """Filtra contornos basándose en ángulos para obtener forma más circular."""
    if not isinstance(contours, list) or len(contours) < 1 or len(contours[0]) < 5:
        return np.array([], dtype=np.int32).reshape((-1, 1, 2))
    
    if len(contours[0].shape) == 2:
        all_contours = contours[0].reshape((-1, 1, 2))
    else:
        all_contours = contours[0]
    
    spacing = max(1, int(len(all_contours) / 25))
    filtered_points = []
    centroid = np.mean(all_contours, axis=0).reshape(2)
    
    for i in range(len(all_contours)):
        current_point = all_contours[i].reshape(2)
        prev_point = all_contours[i - spacing].reshape(2)
        next_point = all_contours[(i + spacing) % len(all_contours)].reshape(2)
        
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            if norm_vec1 == 0 or norm_vec2 == 0:
                continue
        
        vec_to_centroid = centroid - current_point
        if np.dot(vec_to_centroid, (vec1 + vec2)) > 0:
            filtered_points.append(all_contours[i])
    
    if not filtered_points or len(filtered_points) < 5:
        return np.array([], dtype=np.int32).reshape((-1, 1, 2))
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

# ==========================================
# FUNCIONES UTILITARIAS
# ==========================================

def update_and_average_point(point_list, new_point, N):
    """Mantiene promedio móvil de últimos N puntos."""
    point_list.append(new_point)
    if len(point_list) > N:
        point_list.pop(0)
    if not point_list:
        return None
    
    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))
    return (avg_x, avg_y)

def find_line_intersection(ellipse1, ellipse2):
    """Calcula intersección entre dos líneas definidas por elipses."""
    try:
        (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
        (cx2, cy2), (_, minor_axis2), angle2 = ellipse2
        
        if minor_axis1 <= 0 or minor_axis2 <= 0:
            return None
        
        angle1_rad = np.deg2rad(angle1)
        angle2_rad = np.deg2rad(angle2)
        
        dx1 = (minor_axis1 / 2.0) * np.cos(angle1_rad)
        dy1 = (minor_axis1 / 2.0) * np.sin(angle1_rad)
        dx2 = (minor_axis2 / 2.0) * np.cos(angle2_rad)
        dy2 = (minor_axis2 / 2.0) * np.sin(angle2_rad)
        
        A = np.array([[dx1, -dx2], [dy1, -dy2]])
        B = np.array([cx2 - cx1, cy2 - cy1])
        
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-6:
            return None
        
        solution = np.linalg.solve(A, B)
        t1 = solution[0]
        
        intersection_x = cx1 + t1 * dx1
        intersection_y = cy1 + t1 * dy1
        
        return (int(round(intersection_x)), int(round(intersection_y)))
    
    except (ValueError, TypeError, np.linalg.LinAlgError, IndexError):
        return None

def compute_average_intersection(frame, ray_lines, N, M, spacing, current_center_avg):
    """Calcula centro promedio del modelo esférico del ojo."""
    if not hasattr(compute_average_intersection, 'stored_intersections'):
        compute_average_intersection.stored_intersections = []
    
    stored_intersections = compute_average_intersection.stored_intersections
    
    if len(ray_lines) < 2 or N < 2:
        return None
    
    height, width = frame.shape[:2]
    num_to_sample = min(N, len(ray_lines))
    selected_lines = random.sample(ray_lines, num_to_sample)
    
    for i in range(len(selected_lines) - 1):
        line1, line2 = selected_lines[i], selected_lines[i + 1]
        
        if not isinstance(line1, (tuple, list)) or len(line1) != 3 or \
           not isinstance(line2, (tuple, list)) or len(line2) != 3:
            continue
        
        try:
            angle1, angle2 = line1[2], line2[2]
        except IndexError:
            continue
        
        if abs(angle1 - angle2) >= 2.0:
            intersection = find_line_intersection(line1, line2)
            if intersection:
                ix, iy = intersection
                if (0 <= ix < width) and (0 <= iy < height):
                    dist = math.hypot(ix - current_center_avg[0], 
                                     iy - current_center_avg[1])
                    if dist < MAX_INTERSECTION_DISTANCE:
                        stored_intersections.append(intersection)
    
    if len(stored_intersections) > M:
        compute_average_intersection.stored_intersections = stored_intersections[-M:]
    
    current_history = compute_average_intersection.stored_intersections
    if not current_history:
        return None
    
    avg_x = np.mean([pt[0] for pt in current_history])
    avg_y = np.mean([pt[1] for pt in current_history])
    return (int(avg_x), int(avg_y))

def compute_gaze_vector(x_pupil, y_pupil, x_sphere, y_sphere, max_radius_pixels, 
                       screen_width=640, screen_height=480):
    """Calcula vector de mirada 3D desde posición 2D de pupila."""
    try:
        # Centro de esfera en coordenadas normalizadas
        sphere_offset_x = (float(x_sphere) / screen_width) * 2.0 - 1.0
        sphere_offset_y = 1.0 - (float(y_sphere) / screen_height) * 2.0
        sphere_center_3d = np.array([sphere_offset_x * 1.5, 
                                     sphere_offset_y * 1.5, 
                                     0.0])
        
        # Diferencia pupila-esfera
        dx = float(x_pupil) - float(x_sphere)
        dy = float(y_pupil) - float(y_sphere)
        
        if max_radius_pixels <= 0:
            max_radius_pixels = 1.0
        
        # Normalizar a componentes X, Y
        gaze_x = dx / max_radius_pixels
        gaze_y = -dy / max_radius_pixels
        
        # Asegurar que está en esfera unitaria
        mag_sq_2d = gaze_x**2 + gaze_y**2
        if mag_sq_2d > 1.0:
            mag_2d = np.sqrt(mag_sq_2d)
            gaze_x /= mag_2d
            gaze_y /= mag_2d
            mag_sq_2d = 1.0
        
        # Calcular componente Z (profundidad)
        gaze_z_sq = max(0.0, 1.0 - mag_sq_2d)
        gaze_z = -np.sqrt(gaze_z_sq)
        
        # Vector unitario
        gaze_direction_3d = np.array([gaze_x, gaze_y, gaze_z])
        norm = np.linalg.norm(gaze_direction_3d)
        
        if norm < 1e-6:
            return sphere_center_3d, np.array([0.0, 0.0, -1.0])
        
        gaze_direction_3d /= norm
        
        return sphere_center_3d, gaze_direction_3d
    
    except Exception as e:
        fallback_center = np.array([0.0, 0.0, 0.0])
        fallback_gaze = np.array([0.0, 0.0, -1.0])
        return fallback_center, fallback_gaze

# ==========================================
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# ==========================================

def detect_pupil_with_yolo(frame):
    """
    Detecta la pupila usando YOLO + binarización en ROI.
    Retorna: (center_x, center_y, ellipse, contour_area) o (None, None, None, 0)
    """
    global model
    
    if model is None:
        return None, None, None, 0
    
    h_frame, w_frame = frame.shape[:2]
    
    # 1. Ejecutar YOLO
    results = model.track(frame, persist=True, verbose=False)
    
    best_box = None
    max_conf = 0.0
    
    if results[0].boxes:
        for box in results[0].boxes:
            if box.conf[0] > max_conf:
                max_conf = box.conf[0]
                best_box = box
    
    if best_box is None:
        return None, None, None, 0
    
    # 2. Obtener y expandir ROI
    x1_raw, y1_raw, x2_raw, y2_raw = best_box.xyxy[0].cpu().numpy().astype(int)
    
    x1 = max(0, x1_raw - YOLO_ROI_EXPANSION_PX)
    y1 = max(0, y1_raw - YOLO_ROI_EXPANSION_PX)
    x2 = min(w_frame, x2_raw + YOLO_ROI_EXPANSION_PX)
    y2 = min(h_frame, y2_raw + YOLO_ROI_EXPANSION_PX)
    
    if x1 >= x2 or y1 >= y2:
        return None, None, None, 0
    
    # 3. Recortar ROI y binarizar
    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, PUPIL_FIXED_THRESHOLD, 255, 
                                  cv2.THRESH_BINARY_INV)
    
    # 4. Encontrar contornos
    contours_roi, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_roi:
        return None, None, None, 0
    
    # 5. Contorno más grande
    best_contour_roi = max(contours_roi, key=cv2.contourArea)
    contour_area = cv2.contourArea(best_contour_roi)
    
    # Traducir a coordenadas absolutas
    best_contour = best_contour_roi + (x1, y1)
    
    # 6. Ajustar elipse
    optimized_contour = optimize_contours_by_angle([best_contour])
    ellipse = None
    
    try:
        if len(optimized_contour) >= 5:
            ellipse = cv2.fitEllipse(optimized_contour)
        elif len(best_contour) >= 5:
            ellipse = cv2.fitEllipse(best_contour)
    except cv2.error:
        ellipse = None
    
    if ellipse is None:
        return None, None, None, 0
    
    # 7. Centro de la pupila
    center_x_raw, center_y_raw = map(int, ellipse[0])
    
    return center_x_raw, center_y_raw, ellipse, contour_area

# ==========================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ==========================================

def process_frames(frame, gray_frame_clahe):
    """
    Procesa un frame y retorna diccionario con datos de detección.
    Incluye información sobre pérdida de detección para detector de parpadeo.
    """
    global ray_lines, max_rays, prev_model_center_avg, max_observed_distance
    global stable_pupil_centers, model_centers
    global last_known_pupil_center, frames_since_last_good_detection
    
    data_dict = {
        "valid_deteccion": False,
        "sphere_center_x": None,
        "sphere_center_y": None,
        "sphere_center_z": None,
        "pupil_center_x": None,
        "pupil_center_y": None,
        "gaze_x": None,
        "gaze_y": None,
        "gaze_z": None,
        "ellipse_width": None,
        "ellipse_height": None,
        "ellipse_angle": None,
        "contour_area": None,
        "frames_lost": frames_since_last_good_detection,  # Para detector de parpadeo
    }
    
    h_frame, w_frame = frame.shape[:2]
    
    # 1. DETECTAR PUPILA
    center_x_raw, center_y_raw, ellipse, contour_area = detect_pupil_with_yolo(frame)
    
    if ellipse is None:
        frames_since_last_good_detection += 1
        
        # Calcular centro del modelo
        model_center_average = prev_model_center_avg
        model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5, 
                                                    model_center_average)
        if model_center is not None:
            model_center_average = update_and_average_point(model_centers, 
                                                           model_center, 800)
            prev_model_center_avg = model_center_average
        
        data_dict["sphere_center_x"] = model_center_average[0]
        data_dict["sphere_center_y"] = model_center_average[1]
        
        return data_dict
    
    # 2. SUAVIZAR CENTRO DE PUPILA
    data_dict["contour_area"] = contour_area
    stable_pupil_center = update_and_average_point(stable_pupil_centers, 
                                                   (center_x_raw, center_y_raw), N=2)
    center_x, center_y = stable_pupil_center if stable_pupil_center else (center_x_raw, center_y_raw)
    
    # 3. FILTRO TEMPORAL (ANTI-PARPADEO)
    new_pupil_center = (center_x, center_y)
    is_detection_temporally_stable = False
    
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
    
    # 4. CALCULAR CENTRO DEL MODELO (ESFERA)
    model_center_average = prev_model_center_avg
    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5, 
                                                model_center_average)
    if model_center is not None:
        model_center_average = update_and_average_point(model_centers, model_center, 800)
        prev_model_center_avg = model_center_average
    
    data_dict["sphere_center_x"] = model_center_average[0]
    data_dict["sphere_center_y"] = model_center_average[1]
    
    # 5. VALIDAR Y CALCULAR VECTOR DE MIRADA
    if is_detection_temporally_stable:
        dist_from_sphere_center = math.hypot(center_x - model_center_average[0],
                                            center_y - model_center_average[1])
        
        if dist_from_sphere_center <= max_observed_distance:
            # Agregar a historial
            ray_lines.append(ellipse)
            if len(ray_lines) > max_rays:
                ray_lines.pop(0)
            
            # Calcular vector de mirada
            center_3d, direction_3d = compute_gaze_vector(
                center_x, center_y,
                model_center_average[0], model_center_average[1],
                max_observed_distance
            )
            
            if center_3d is not None and direction_3d is not None:
                data_dict["valid_deteccion"] = True
                data_dict["sphere_center_z"] = center_3d[2]
                data_dict["pupil_center_x"] = center_x
                data_dict["pupil_center_y"] = center_y
                data_dict["gaze_x"] = direction_3d[0]
                data_dict["gaze_y"] = direction_3d[1]
                data_dict["gaze_z"] = direction_3d[2]
                data_dict["ellipse_width"] = ellipse[1][0]
                data_dict["ellipse_height"] = ellipse[1][1]
                data_dict["ellipse_angle"] = ellipse[2]
    
    return data_dict

def process_frame(frame):
    """
    Función de entrada principal para procesar un frame.
    Retorna diccionario con datos de detección.
    """
    # Recortar a aspecto ratio
    frame_cropped = crop_to_aspect_ratio(frame)
    
    # Preprocesar
    gray_frame = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
    gray_frame_blurred = cv2.GaussianBlur(gray_frame, GAUSSIAN_KERNEL_SIZE, 0)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    gray_frame_clahe = clahe.apply(gray_frame_blurred)
    
    # Procesar
    data_dict = process_frames(frame_cropped, gray_frame_clahe)
    
    return data_dict

# ==========================================
# FUNCIÓN DE RESET
# ==========================================

def reset_tracker_state():
    """Reinicia todas las variables globales del tracker."""
    global ray_lines, model_centers, stable_pupil_centers
    global last_known_pupil_center, frames_since_last_good_detection
    global prev_model_center_avg
    
    ray_lines = []
    model_centers = []
    stable_pupil_centers = []
    prev_model_center_avg = (280, 150)
    last_known_pupil_center = None
    frames_since_last_good_detection = 0
    
    # Limpiar historial de intersecciones
    if hasattr(compute_average_intersection, 'stored_intersections'):
        compute_average_intersection.stored_intersections = []
    
    print("✓ Estado del tracker reiniciado")