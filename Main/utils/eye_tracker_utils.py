import cv2
import random
import math
import numpy as np
import os
import time
import csv

# --- PARÁMETROS DE FILTRADO Y PREPROCESAMIENTO (ACTUALIZADOS) ---
FIXED_THRESHOLD_VALUE = 75   # Umbral fijo (de debug_step_5)
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0       # Límite de clip (de debug_step_5)
MIN_PUPIL_AREA = 1000        # (de debug_step_5)
MAX_PUPIL_AREA = 10000       # (de debug_step_5)
MORPH_KERNEL_SIZE = 5        # (N=2 -> (2*2)+1 = 5, de debug_step_5)
# ------------------------------------------

# --- PARÁMETRO DE ESTABILIDAD DEL MODELO ---
MAX_INTERSECTION_DISTANCE = 55 # Distancia máxima en píxeles
# ------------------------------------------

# --- VARIABLES DE ESTADO GLOBALES ---
ray_lines = []
model_centers = []
stable_pupil_centers = []
max_rays = 120
prev_model_center_avg = (320, 240)
max_observed_distance = 250

# --- FUNCIONES DE PROCESAMIENTO ---

def crop_to_aspect_ratio(image, width=640, height=480):
    """Recorta y reescala la imagen a la relación de aspecto deseada."""
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # La imagen es demasiado ancha
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        # La imagen es demasiado alta
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))

def apply_fixed_binary_threshold(image, threshold_value):
    """Aplica umbral binario INVERSO."""
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# --- FUNCIÓN DE PUNTO MÁS OSCURO (AÑADIDA DE DEBUG_STEP_5) ---
def find_darkest_2x2(image):
    """
    Encuentra el bloque de 2x2 píxeles más oscuro en la imagen.
    Devuelve la coordenada (x, y) de la esquina superior izquierda de ese bloque.
    """
    min_sum = 1021 
    min_loc = (0, 0) # (x, y)
    H, W = image.shape
    
    for y in range(H - 1):
        for x in range(W - 1):
            s = int(image[y, x])     + int(image[y+1, x]) + \
                int(image[y, x+1])   + int(image[y+1, x+1])
            
            if s < min_sum:
                min_sum = s
                min_loc = (x, y)
                
    return min_loc

# --- FUNCIÓN DE OPTIMIZACIÓN DE ÁNGULO (REEMPLAZADA CON DEBUG_STEP_5) ---
def optimize_contours_by_angle(contours):
    """Filtra los puntos de un contorno basándose en el ángulo y la convexidad."""
    if not isinstance(contours, list) or len(contours) < 1 or len(contours[0]) < 5:
        return np.array([], dtype=np.int32).reshape((-1, 1, 2))

    # Asegurarse de que el contorno tenga la forma correcta (N, 1, 2)
    if len(contours[0].shape) == 2:
        all_contours = contours[0].reshape((-1, 1, 2))
    else:
        all_contours = contours[0]

    spacing = max(1, int(len(all_contours)/25))
    filtered_points = []
    
    # Calcular centroide
    centroid = np.mean(all_contours, axis=0).reshape(2)
    
    for i in range(0, len(all_contours)):
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
                
            dot_product = np.dot(vec1, vec2)
            # Evitar valores fuera de rango para arccos
            dot_product = np.clip(dot_product / (norm_vec1 * norm_vec2), -1.0, 1.0)
            angle = np.arccos(dot_product)

        vec_to_centroid = centroid - current_point
        
        if np.dot(vec_to_centroid, (vec1+vec2)) > 0: # Comprueba si apunta 'hacia adentro'
            filtered_points.append(all_contours[i])
    
    if not filtered_points or len(filtered_points) < 5:
        return np.array([], dtype=np.int32).reshape((-1, 1, 2))

    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))


# --- FUNCIÓN 'detect_and_select_pupil' ELIMINADA ---
# La nueva lógica está integrada directamente en 'process_frames'


def process_frames(frame, gray_frame_clahe):
    """
    Procesa un frame y DEVUELVE un diccionario con datos, incluyendo el área del contorno.
    (LÓGICA CENTRAL ACTUALIZADA)
    """
    global ray_lines, max_rays, prev_model_center_avg, max_observed_distance, stable_pupil_centers, model_centers

    # Añadir 'contour_area' al diccionario
    data_dict = {
        "valid_deteccion": False, "sphere_center_x": None, "sphere_center_y": None, "sphere_center_z": None,
        "pupil_center_x": None, "pupil_center_y": None,
        "darkest_pixel_x": None, "darkest_pixel_y": None, # (Ahora se poblará con el anchor 2x2)
        "gaze_x": None, "gaze_y": None, "gaze_z": None,
        "ellipse_width": None, "ellipse_height": None, "ellipse_angle": None,
        "contour_area": None
    }

    # --- INICIO DE LA NUEVA LÓGICA (DE DEBUG_STEP_5) ---

    # 1. Encontrar punto ancla (el más oscuro)
    dark_point = find_darkest_2x2(gray_frame_clahe) # (x, y)
    
    # 2. Binarización y Morfología
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    
    thresholded_image_raw = apply_fixed_binary_threshold(gray_frame_clahe, FIXED_THRESHOLD_VALUE)
    thresholded_image_closed = cv2.morphologyEx(thresholded_image_raw, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    thresholded_image_final = cv2.morphologyEx(thresholded_image_closed, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # 3. Filtrado de Contorno por Punto Ancla
    contours, _ = cv2.findContours(thresholded_image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pupil_contour = None
    for contour in contours:
        
        # Filtro 1: Área (de debug_step_5)
        contour_area = cv2.contourArea(contour)
        if not (MIN_PUPIL_AREA <= contour_area <= MAX_PUPIL_AREA):
            continue # Descarta si es muy pequeño o muy grande

        # Filtro 2: Punto Ancla (de debug_step_5)
        if cv2.pointPolygonTest(contour, dark_point, False) >= 0:
            pupil_contour = contour
            data_dict["contour_area"] = contour_area # Guardar el área del contorno válido
            break # Encontramos el que queríamos

    # --- FIN DE LA NUEVA LÓGICA ---

    final_rotated_rect = None
    center_x, center_y = None, None

    if pupil_contour is not None:
        # (El área ya se guardó en el bucle anterior)

        # Guardar el ANCLA 2x2 como el "píxel más oscuro"
        data_dict["darkest_pixel_x"] = dark_point[0]
        data_dict["darkest_pixel_y"] = dark_point[1]

        # Optimizar contorno (de debug_step_5)
        optimized_contour = optimize_contours_by_angle([pupil_contour])

        # Ajustar elipse final
        ellipse = None
        try:
            if len(optimized_contour) >= 5:
                ellipse = cv2.fitEllipse(optimized_contour)
            else:
                 # Fallback al contorno original si la optimización eliminó demasiados puntos
                 if len(pupil_contour) >= 5:
                     ellipse = cv2.fitEllipse(pupil_contour)
        except cv2.error:
            ellipse = None # Manejar errores de ajuste

        if ellipse is not None:
            final_rotated_rect = ellipse
            center_x_raw, center_y_raw = map(int, final_rotated_rect[0])

            # Suavizado de la posición (centro de la elipse)
            stable_pupil_center = update_and_average_point(stable_pupil_centers, (center_x_raw, center_y_raw), N=3)
            if stable_pupil_center:
                center_x, center_y = stable_pupil_center # Usar centro suavizado
            else:
                 center_x, center_y = center_x_raw, center_y_raw # Usar crudo si no hay historial

            # Almacenar rayos
            ray_lines.append(final_rotated_rect)
            if len(ray_lines) > max_rays:
                ray_lines = ray_lines[-max_rays:]

    # Calcular centro del modelo (Esfera Cian) - Se calcula siempre
    model_center_average = prev_model_center_avg
    # Pasar el promedio actual como referencia para el filtro de distancia
    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5, model_center_average)
    
    if model_center is not None:
        model_center_average = update_and_average_point(model_centers, model_center, 800)
        prev_model_center_avg = model_center_average # Actualizar referencia para el siguiente frame

    # Poblar coordenadas del centro de la esfera (siempre existen)
    data_dict["sphere_center_x"] = model_center_average[0]
    data_dict["sphere_center_y"] = model_center_average[1]

    # Dibujar y poblar el resto de datos si hubo detección válida este frame
    if final_rotated_rect is not None and center_x is not None:
        # Dibujar elipse y líneas
        cv2.ellipse(frame, final_rotated_rect, (0, 255, 255), 2) # Elipse amarilla
        cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2) # Línea de mirada
        dx = center_x - model_center_average[0]
        dy = center_y - model_center_average[1]
        ex = int(model_center_average[0] + 2 * dx)
        ey = int(model_center_average[1] + 2 * dy)
        cv2.line(frame, (center_x, center_y), (ex, ey), (200, 255, 0), 3) # Línea extendida

        # Dibujar punto ancla 2x2 (Punto rojo, de debug_step_5)
        if data_dict["darkest_pixel_x"] is not None:
             dark_pix_coords = (data_dict["darkest_pixel_x"], data_dict["darkest_pixel_y"])
             # Dibujamos un círculo centrado en la esquina (x,y) del ancla 2x2
             #cv2.circle(frame, dark_pix_coords, 5, (0, 0, 255), -1) 

        # Calcular vector de mirada 3D
        center_3d, direction_3d = compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1])

        if center_3d is not None and direction_3d is not None:
            # Poblar el diccionario completamente
            data_dict["valid_deteccion"] = True
            data_dict["sphere_center_z"] = center_3d[2] # Z viene del cálculo 3D
            data_dict["pupil_center_x"] = center_x # Centro 2D suavizado de la elipse
            data_dict["pupil_center_y"] = center_y
            data_dict["gaze_x"] = direction_3d[0]
            data_dict["gaze_y"] = direction_3d[1]
            data_dict["gaze_z"] = direction_3d[2]
            data_dict["ellipse_width"] = final_rotated_rect[1][0]
            data_dict["ellipse_height"] = final_rotated_rect[1][1]
            data_dict["ellipse_angle"] = final_rotated_rect[2]

            # Dibujar texto informativo
            origin_text = f"Origin: ({center_3d[0]:.2f}, {center_3d[1]:.2f}, {center_3d[2]:.2f})"
            dir_text = f"Direction: ({direction_3d[0]:.2f}, {direction_3d[1]:.2f}, {direction_3d[2]:.2f})"
            cv2.putText(frame, origin_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, dir_text, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Dibujar modelo del ojo (siempre)
    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)

    # Mostrar el frame procesado
    cv2.imshow("Frame with Ellipse and Rays", frame)

    return data_dict


def update_and_average_point(point_list, new_point, N):
    """Añade un punto a una lista y calcula el promedio de los últimos N puntos."""
    point_list.append(new_point)
    # Si la lista excede N, elimina el más antiguo
    if len(point_list) > N:
        point_list.pop(0)
    # Si no hay puntos, no se puede promediar
    if not point_list:
        return None
    # Calcular promedio y redondear a entero
    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))
    return (avg_x, avg_y)


def compute_average_intersection(frame, ray_lines, N, M, spacing, current_center_avg):
    """Calcula el punto promedio de intersección de elipses recientes."""
    # Usar atributo de función para almacenamiento persistente
    if not hasattr(compute_average_intersection, 'stored_intersections'):
        compute_average_intersection.stored_intersections = []
    stored_intersections = compute_average_intersection.stored_intersections

    if len(ray_lines) < 2 or N < 2:
        return None # No suficientes líneas para calcular

    height, width = frame.shape[:2]
    num_to_sample = min(N, len(ray_lines))
    selected_lines = random.sample(ray_lines, num_to_sample)

    new_intersections_this_frame = []

    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]

        # Validar datos de elipse antes de desempaquetar
        if not isinstance(line1, (tuple, list)) or len(line1) != 3 or \
           not isinstance(line2, (tuple, list)) or len(line2) != 3:
            continue

        try:
            angle1 = line1[2]
            angle2 = line2[2]
        except IndexError:
             continue # Si la tupla no tiene índice 2

        # Filtrar líneas paralelas
        if abs(angle1 - angle2) >= 2.0:
            intersection = find_line_intersection(line1, line2)

            if intersection:
                ix, iy = intersection
                # Comprobar si está dentro de los límites del frame
                if (0 <= ix < width) and (0 <= iy < height):
                    # Comprobar distancia desde el centro promedio ACTUAL
                    dist = math.hypot(ix - current_center_avg[0], iy - current_center_avg[1])

                    if dist < MAX_INTERSECTION_DISTANCE: # Aplicar filtro
                        new_intersections_this_frame.append(intersection)
                        stored_intersections.append(intersection) # Añadir a la lista persistente

    # Limitar el historial de intersecciones
    if len(stored_intersections) > M:
        compute_average_intersection.stored_intersections = stored_intersections[-M:]

    # Calcular promedio usando el historial actualizado
    current_history = compute_average_intersection.stored_intersections
    if not current_history:
        return None # No hay intersecciones válidas

    avg_x = np.mean([pt[0] for pt in current_history])
    avg_y = np.mean([pt[1] for pt in current_history])

    return (int(avg_x), int(avg_y))


def find_line_intersection(ellipse1, ellipse2):
    """Encuentra el punto de intersección de las líneas definidas por las elipses."""
    try:
        (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
        (cx2, cy2), (_, minor_axis2), angle2 = ellipse2

        # Validar ejes menores
        if minor_axis1 <= 0 or minor_axis2 <= 0:
            return None

        angle1_rad = np.deg2rad(angle1)
        angle2_rad = np.deg2rad(angle2)

        # Vectores de dirección basados en eje menor y ángulo
        dx1 = (minor_axis1 / 2.0) * np.cos(angle1_rad)
        dy1 = (minor_axis1 / 2.0) * np.sin(angle1_rad)
        dx2 = (minor_axis2 / 2.0) * np.cos(angle2_rad)
        dy2 = (minor_axis2 / 2.0) * np.sin(angle2_rad)

        # Sistema lineal: A * [t1, -t2]^T = B
        A = np.array([[dx1, -dx2], [dy1, -dy2]])
        B = np.array([cx2 - cx1, cy2 - cy1])

        # Comprobar determinante para líneas paralelas
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-6: # Usar tolerancia
            return None

        # Resolver para t1
        solution = np.linalg.solve(A, B)
        t1 = solution[0]

        # Calcular punto de intersección
        intersection_x = cx1 + t1 * dx1
        intersection_y = cy1 + t1 * dy1

        # Redondear antes de convertir a entero
        return (int(round(intersection_x)), int(round(intersection_y)))

    except (ValueError, TypeError, np.linalg.LinAlgError, IndexError):
         # Capturar errores de desempaquetado, matemáticos o de álgebra lineal
        return None


def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    """Calcula el vector de mirada 3D."""
    try:
        # Parámetros de cámara y proyección
        viewport_width = float(screen_width)
        viewport_height = float(screen_height)
        fov_y_deg = 45.0
        aspect_ratio = viewport_width / viewport_height
        far_clip = 100.0
        camera_position = np.array([0.0, 0.0, 3.0])

        # Coordenadas 2D a NDC
        ndc_x = (2.0 * float(x)) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * float(y)) / viewport_height

        # Punto 3D en el plano lejano
        fov_y_rad = np.radians(fov_y_deg)
        half_height_far = np.tan(fov_y_rad / 2.0) * far_clip
        half_width_far = half_height_far * aspect_ratio
        far_x = ndc_x * half_width_far
        far_y = ndc_y * half_height_far
        far_z = camera_position[2] - far_clip
        far_point = np.array([far_x, far_y, far_z])

        # Dirección del rayo desde la cámara
        ray_origin = camera_position
        ray_direction_world = far_point - ray_origin
        norm_direction = np.linalg.norm(ray_direction_world)
        if norm_direction == 0: return None, None
        ray_direction_world /= norm_direction

        # Modelo simplificado de esfera ocular
        inner_radius = 1.0 / 1.05
        sphere_offset_x = (float(center_x) / screen_width) * 2.0 - 1.0
        sphere_offset_y = 1.0 - (float(center_y) / screen_height) * 2.0
        sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

        # Intersección Rayo-Esfera
        origin = ray_origin
        direction = ray_direction_world # Dirección correcta
        L = origin - sphere_center
        a = np.dot(direction, direction) # Debería ser ~1.0
        b = 2.0 * np.dot(direction, L)
        c = np.dot(L, L) - inner_radius**2
        discriminant = b**2 - 4.0 * a * c

        if discriminant < 0:
            return sphere_center, np.array([0.0, 0.0, -1.0]) # Fallback: mirar hacia adelante

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # Encontrar la intersección válida más cercana (t > 0)
        t = None
        if t1 > 1e-6 and t2 > 1e-6:
            t = min(t1, t2)
        elif t1 > 1e-6:
            t = t1
        elif t2 > 1e-6:
            t = t2

        if t is None: # Intersecciones detrás de la cámara
            return sphere_center, np.array([0.0, 0.0, -1.0])

        intersection_point = origin + t * direction

        # Vector de mirada: del centro de la esfera al punto de intersección
        gaze_direction = intersection_point - sphere_center
        norm_gaze = np.linalg.norm(gaze_direction)
        if norm_gaze == 0:
            return sphere_center, np.array([0.0, 0.0, -1.0]) # Fallback
        gaze_direction /= norm_gaze

        return sphere_center, gaze_direction

    except Exception as e:
         # print(f"Error en compute_gaze_vector: {e}")
         return None, None


def process_frame(frame):
    """Procesa un solo frame: Preprocesa, aplica CLAHE, detecta, calcula y devuelve datos."""
    # 1. Recorte
    frame = crop_to_aspect_ratio(frame)
    # 2. Escala de grises
    gray_frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 3. Desenfoque Gaussiano
    gray_frame_blurred = cv2.GaussianBlur(gray_frame_original, GAUSSIAN_KERNEL_SIZE, 0)
    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    gray_frame_clahe = clahe.apply(gray_frame_blurred)
    # 5. Procesamiento principal y obtención de datos
    data_dict = process_frames(frame, gray_frame_clahe)
    return data_dict


def process_video_from_path(video_path, video_name, csv_path):
    """
    Abre y procesa un archivo de video, guarda CSV y imprime min/max área del contorno.
    """
    global ray_lines, model_centers, stable_pupil_centers, prev_model_center_avg

    # Resetear estado global para el nuevo video
    ray_lines, model_centers, stable_pupil_centers = [], [], []
    prev_model_center_avg = (320, 240)
    if hasattr(compute_average_intersection, 'stored_intersections'):
        compute_average_intersection.stored_intersections = []

    # Inicializar variables para min/max área
    min_area_found = float('inf')
    max_area_found = float('-inf')

    # Abrir video
    if not os.path.exists(video_path):
        print(f"Error: Archivo de video no encontrado en {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video {video_path}")
        return

    # Obtener FPS y calcular delay
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0 # Default a 30 FPS si falla
    frame_delay = int(1000 / fps)
    frame_counter = 0

    print(f"Procesando video: {video_path}\nGuardando datos CSV en: {csv_path}\nPresiona 'q' para salir, 'espacio' para pausar")

    # Encabezado CSV
    csv_header = [
        "video_name", "frame_number", "timestamp_ms", "valid_deteccion",
        "sphere_center_x", "sphere_center_y", "sphere_center_z",
        "pupil_center_x", "pupil_center_y",
        "darkest_pixel_x", "darkest_pixel_y", # (Ahora guardará el ancla 2x2)
        "gaze_x", "gaze_y", "gaze_z",
        "ellipse_width", "ellipse_height", "ellipse_angle",
        "contour_area"
    ]

    try:
        # Abrir archivo CSV para escritura
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header) # Escribir encabezado

            # Bucle principal de procesamiento de frames
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret: # Si no hay más frames, salir del bucle
                    break

                frame_counter += 1
                timestamp_ms = (frame_counter / fps) * 1000.0

                # Procesar el frame actual
                data = process_frame(frame)

                # Actualizar min/max área si la detección fue válida
                if data.get("valid_deteccion") and data.get("contour_area") is not None:
                    current_area = data["contour_area"]
                    min_area_found = min(min_area_found, current_area)
                    max_area_found = max(max_area_found, current_area)

                # Preparar la fila para el CSV (manejo de None)
                row = [
                    video_name, frame_counter, f"{timestamp_ms:.3f}", data.get("valid_deteccion", False),
                    f"{data.get('sphere_center_x', ''):.3f}" if data.get('sphere_center_x') is not None else '',
                    f"{data.get('sphere_center_y', ''):.3f}" if data.get('sphere_center_y') is not None else '',
                    f"{data.get('sphere_center_z', ''):.3f}" if data.get('sphere_center_z') is not None else '',
                    f"{data.get('pupil_center_x', ''):.3f}" if data.get('pupil_center_x') is not None else '',
                    f"{data.get('pupil_center_y', ''):.3f}" if data.get('pupil_center_y') is not None else '',
                    data.get('darkest_pixel_x', ''),
                    data.get('darkest_pixel_y', ''),
                    f"{data.get('gaze_x', ''):.6f}" if data.get('gaze_x') is not None else '',
                    f"{data.get('gaze_y', ''):.6f}" if data.get('gaze_y') is not None else '',
                    f"{data.get('gaze_z', ''):.6f}" if data.get('gaze_z') is not None else '',
                    f"{data.get('ellipse_width', ''):.3f}" if data.get('ellipse_width') is not None else '',
                    f"{data.get('ellipse_height', ''):.3f}" if data.get('ellipse_height') is not None else '',
                    f"{data.get('ellipse_angle', ''):.3f}" if data.get('ellipse_angle') is not None else '',
                    f"{data.get('contour_area', ''):.1f}" if data.get('contour_area') is not None else ''
                ]
                writer.writerow(row) # Escribir la fila

                # Control de teclado y delay para visualización
                processing_duration_ms = (time.time() - start_time) * 1000
                wait_time = max(1, frame_delay - int(processing_duration_ms))
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'):
                    print("Procesamiento detenido por el usuario.")
                    break
                elif key == ord(' '):
                    print("Pausado. Presiona cualquier tecla para continuar...")
                    cv2.waitKey(0) # Pausa indefinida

    except IOError as e:
        print(f"Error escribiendo en el archivo CSV {csv_path}: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante el procesamiento: {e}")
        # Considerar añadir traceback para depuración más detallada:
        # import traceback
        # traceback.print_exc()
    finally:
        # Asegurarse de liberar recursos y cerrar ventanas
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

        # Imprimir resumen de áreas al finalizar el video
        print(f"--- [COMPLETADO] Video: {video_name} ---")
        if max_area_found == float('-inf'): # Si no se detectó ninguna pupila
            print("    -> No se detectaron pupilas válidas en este video.")
        else:
            print(f"    -> Área Contorno Mín: {min_area_found:.1f}, Área Contorno Máx: {max_area_found:.1f}")