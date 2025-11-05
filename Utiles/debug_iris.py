import cv2
import numpy as np
import os
import time
import math 
# (Matplotlib eliminado)

# --- PARÁMETROS DE PREPROCESAMIENTO (FIJOS) ---
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0 
FIXED_THRESHOLD_VALUE = 30 
FIXED_KERNEL_N = 2         

# --- PARÁMETROS DE FILTRADO PUPILA (FIJOS) ---
FIXED_MIN_PUPIL_AREA = 500 
FIXED_MAX_PUPIL_AREA = 8000  

# --- PARÁMETRO DE ROBUSTEZ (FILTRO DE FIT ELÍPTICO) ---
MIN_ELLIPTICAL_FIT_RATIO = 0.85  
MAX_ELLIPTICAL_FIT_RATIO = 1.20 
# ----------------------------------------------

# --- PARÁMETRO DE FILTRO DE BBOX PUPILA (FIJO) ---
HORIZONTALITY_TOLERANCE = 1.40 
# -----------------------------------------------

# --- <<<--- PARÁMETROS DE BÚSQUEDA ELÍPTICA (Tus valores) ---
IRIS_RADIAL_RAYS = 450            
POSITIVE_GRADIENT_THRESHOLD = 2   
RING_END_THRESHOLD = -2           
PUPIL_SCALE_START = 1.0           
PUPIL_SCALE_END = 5.2 # (Tu valor)ddddddddddd
NUM_ELLIPTICAL_STEPS = 100        
# --- <<<--- ---

# --- ¡NUEVOS! PARÁMETROS DE FILTRADO (Basado en Radio Normalizado) ---
MIN_NORMALIZED_RADIUS = 1.5    # (Filtro 1a) Mínimo 1.2x el tamaño de la pupila
MAX_NORMALIZED_RADIUS = 4.2   # (Filtro 1b) ¡NUEVO! Máximo 4.5x (antes de 5.0)
ROBUST_FILTER_THRESHOLD = 2    # (Filtro 2) Umbral de pertenencia (Z-score)
MORPH_CLEANUP_KERNEL_SIZE = 7    # (Filtro 3) Kernel morfológico
# --- <<<--- ---

# --- ETAPA 5 (SUAVIZADO) ---
IRIS_SMOOTHING_ALPHA = 0.33 # Alpha para EMA (Equivalente a 5 frames: 2 / (5 + 1))
# --- <<<--- ---

# ATENCIÓN: Ruta actualizada a tu versión de Windows
VIDEO_PATH = r"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/Arteaga/Arteaga_intento_1.mp4" 

# --- FUNCIONES DE UTILIDAD (SIN CAMBIOS) ---
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
    _, thresholded_image = cv2.threshold(image, int(threshold_value), 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# (Función 'optimize_contours_by_angle' corregida)
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

def obtener_oscuridad_media_contorno(image_gray, contour):
    if contour is None or len(contour) == 0:
        return 255.0
    mask = np.zeros(image_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), cv2.FILLED)
    if np.sum(mask) == 0:
        return 255.0 
    mean, stddev = cv2.meanStdDev(image_gray, mask=mask)
    return mean[0][0]

def on_trackbar(val): pass

# --- FUNCIÓN PRINCIPAL DE DEBUG (Modificada) ---
def debug_full_frame_processing(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video: {video_path}"); return
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30
    frame_delay = int(1000 / fps)

    window_name = "Elliptical Search Debug" 
    cv2.namedWindow(window_name)
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

    # --- Variable de estado para el suavizado ---
    smoothed_iris_ellipse = ((0,0),(0,0),0)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        # --- 1. Preprocessing ---
        frame_cropped = crop_to_aspect_ratio(frame)
        h_frame, w_frame = frame_cropped.shape[:2] 
        
        frame_blurred = cv2.GaussianBlur(frame_cropped, GAUSSIAN_KERNEL_SIZE, 0)
        gray_frame_original = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
        gray_frame_clahe = clahe.apply(gray_frame_original) 

        # --- 3. Binarization & Morphology (Pupila) ---
        kernel_size_pupil = (FIXED_KERNEL_N * 2) + 1
        morph_kernel_pupil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_pupil, kernel_size_pupil))
        
        thresholded_image_raw = apply_fixed_binary_threshold(gray_frame_clahe, FIXED_THRESHOLD_VALUE)
        thresholded_image_closed = cv2.morphologyEx(thresholded_image_raw, cv2.MORPH_CLOSE, morph_kernel_pupil, iterations=1)
        thresholded_image_final = cv2.morphologyEx(thresholded_image_closed, cv2.MORPH_OPEN, morph_kernel_pupil, iterations=1)

        # --- 5. Contour Filtering (Pupila) ---
        contours, _ = cv2.findContours(thresholded_image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_in_area_range = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if FIXED_MIN_PUPIL_AREA <= contour_area <= FIXED_MAX_PUPIL_AREA:
                contours_in_area_range.append(contour)

        # --- LÓGICA DE SELECCIÓN (Pupila) ---
        best_pupil_contour = None
        best_fit_score = float('inf') 
        final_ellipse = None 
        pupil_diameter = 0 
        
        good_fit_contours = []
        discarded_horizontal_contours = []

        for contour in contours_in_area_range:
            if len(contour) < 5: continue
            
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(contour)
            if w_bbox > (h_bbox * HORIZONTALITY_TOLERANCE):
                discarded_horizontal_contours.append(contour)
                continue
            
            try:
                fitted_ellipse = cv2.fitEllipse(contour)
                (width, height) = fitted_ellipse[1]
                if width <= 0 or height <= 0: continue
                ellipse_area = (np.pi / 4.0) * width * height
                if ellipse_area <= 0: continue
                contour_area = cv2.contourArea(contour)
                fit_ratio = contour_area / ellipse_area

                if MIN_ELLIPTICAL_FIT_RATIO < fit_ratio <= MAX_ELLIPTICAL_FIT_RATIO:
                    good_fit_contours.append(contour)
                    current_fit_score = abs(fit_ratio - 1.0)
                    
                    if current_fit_score < best_fit_score:
                        best_fit_score = current_fit_score
                        best_pupil_contour = contour
            except cv2.error:
                continue
        
        # --- Lienzo de Debug (Derecha) ---
        final_bgr_display = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)
        
        # --- 6. Optimization & Drawing (Pupila) ---
        if best_pupil_contour is not None:
            optimized_contour = optimize_contours_by_angle([best_pupil_contour])
            try:
                if len(optimized_contour) >= 5:
                    final_ellipse = cv2.fitEllipse(optimized_contour)
                else:
                    final_ellipse = cv2.fitEllipse(best_pupil_contour)
                pupil_diameter = max(final_ellipse[1]) 
            except cv2.error: 
                final_ellipse = None
        
        # --- ############################################# ---
        # --- INICIO: LÓGICA DE BÚSQUEDA ELÍPTICA ---
        # --- ############################################# ---
        
        iris_edge_points = []
        cleaned_points = [] 
        current_fitted_ellipse = None
        
        if final_ellipse is None:
            smoothed_iris_ellipse = ((0,0),(0,0),0)
        
        if final_ellipse:
            
            # --- 1. Dibujar pupila en ambos lienzos ---
            cv2.ellipse(frame_cropped, final_ellipse, (0, 255, 255), 2) # Pupila (Amarillo)
            cv2.ellipse(final_bgr_display, final_ellipse, (0, 100, 0), 2) # Pupila (Verde Oscuro)

            # --- 2. Extraer parámetros de la elipse de pupila ---
            (cx_f, cy_f), (w_axis, h_axis), angle_deg = final_ellipse
            cx, cy = int(cx_f), int(cy_f)
            
            semi_a = w_axis / 2.0
            semi_b = h_axis / 2.0
            if semi_a == 0: semi_a = 1.0
            if semi_b == 0: semi_b = 1.0
            
            angle_rad = np.deg2rad(angle_deg)
            cos_rot = np.cos(angle_rad)
            sin_rot = np.sin(angle_rad)

            # --- 2b. Calcular y dibujar el ÁREA DE BÚSQUEDA EXTERIOR ---
            s_outer = PUPIL_SCALE_END
            outer_axes = (w_axis * s_outer, h_axis * s_outer)
            outer_search_ellipse = ((cx_f, cy_f), outer_axes, angle_deg)
            
            cv2.ellipse(frame_cropped, outer_search_ellipse, (0, 165, 255), 2) # Naranja
            cv2.ellipse(final_bgr_display, outer_search_ellipse, (0, 165, 255), 1) # Naranja (tenue)
            
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
                        try:
                            final_bgr_display[y_curr, x_curr] = (50, 0, 0) # Azul oscuro
                        except IndexError: pass
                    
                    elif state == "READY_TO_PAINT":
                        try:
                            final_bgr_display[y_curr, x_curr] = (50, 0, 0) # Azul oscuro
                        except IndexError: pass

                        if gradient > POSITIVE_GRADIENT_THRESHOLD:
                            try:
                                final_bgr_display[y_curr, x_curr] = (0, 255, 0) # Verde
                                iris_edge_points.append((x_curr, y_curr))
                            except IndexError: pass
                            state = "DONE"

                    elif state == "DONE":
                        break
                        
                    prev_intensity = curr_intensity
                    x_prev, y_prev = x_curr, y_curr

            # --- 5. FILTRO DE TRES ETAPAS (¡LÓGICA REESCRITA!) ---
            if len(iris_edge_points) > 0:
                
                # --- Pre-cálculo: Calcular Radio Normalizado para *todos* los puntos ---
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
                stage_one_inliers_pts = [] # (x, y)
                stage_one_inliers_rad = [] # (radio_normalizado)
                stage_one_outliers_pts = [] # (x, y)
                
                for pt, r_norm in points_data:
                    # ¡LÓGICA MODIFICADA!
                    if r_norm < MIN_NORMALIZED_RADIUS or r_norm > MAX_NORMALIZED_RADIUS:
                        stage_one_outliers_pts.append(pt)
                    else:
                        stage_one_inliers_pts.append(pt)
                        stage_one_inliers_rad.append(r_norm)
                
                # --- ETAPA 2: Filtro Dinámico (Mediana sobre Radio Normalizado) ---
                stage_two_inliers_pts = [] 
                stage_two_outliers_pts = [] 

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
                            stage_two_outliers_pts.append(pt)
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
                    cleaned_points = list(zip(coords[1], coords[0])) # (x, y)

                # --- ETAPA 4: AJUSTE DE ELIPSE ---
                if len(cleaned_points) >= 5:
                    try:
                        final_points_np = np.array(cleaned_points, dtype=np.int32).reshape((-1, 1, 2))
                        current_fitted_ellipse = cv2.fitEllipse(final_points_np) # Elipse "cruda"
                    except cv2.error as e:
                        current_fitted_ellipse = None
                
                # --- ETAPA 5: SUAVIZADO (EMA) ---
                if current_fitted_ellipse is not None:
                    if smoothed_iris_ellipse[1][0] == 0.0: # Si es la primera vez
                        smoothed_iris_ellipse = current_fitted_ellipse
                    else:
                        alpha = IRIS_SMOOTHING_ALPHA
                        scx = (current_fitted_ellipse[0][0] * alpha) + (smoothed_iris_ellipse[0][0] * (1.0 - alpha))
                        scy = (current_fitted_ellipse[0][1] * alpha) + (smoothed_iris_ellipse[0][1] * (1.0 - alpha))
                        sax = (current_fitted_ellipse[1][0] * alpha) + (smoothed_iris_ellipse[1][0] * (1.0 - alpha))
                        say = (current_fitted_ellipse[1][1] * alpha) + (smoothed_iris_ellipse[1][1] * (1.0 - alpha))
                        sang = (current_fitted_ellipse[2] * alpha) + (smoothed_iris_ellipse[2] * (1.0 - alpha))
                        smoothed_iris_ellipse = ((scx, scy), (sax, say), sang)

                # --- 6. DIBUJAR LOS PUNTOS ---
                for pt in stage_one_outliers_pts:
                    cv2.circle(final_bgr_display, pt, 2, (0, 0, 255), -1) # Outliers E1 (Rojo)
                for pt in stage_two_outliers_pts:
                    cv2.circle(final_bgr_display, pt, 2, (0, 255, 255), -1) # Outliers E2 (Amarillo)
                for pt in cleaned_points:
                    cv2.circle(final_bgr_display, pt, 2, (255, 255, 255), -1) # Inliers Finales (Blanco)

        # --- ############################################# ---
        # --- FIN: LÓGICA DE BÚSQUEDA ELÍPTICA ---
        # --- ############################################# ---

        # --- 7. Final Visualization ---
        
        # Dibujar la elipse SUAVIZADA
        if smoothed_iris_ellipse[1][0] > 0:
            cv2.ellipse(frame_cropped, smoothed_iris_ellipse, (255, 0, 255), 2) # Iris (Magenta)
            cv2.ellipse(final_bgr_display, smoothed_iris_ellipse, (255, 0, 255), 2) # Iris (Magenta)
        
        # Añadir textos a la vista del frame
        cv2.putText(frame_cropped, f"Pupil Diameter: {pupil_diameter:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_cropped, f"Pos Grad: {POSITIVE_GRADIENT_THRESHOLD}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_cropped, f"Ring End: {RING_END_THRESHOLD}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_cropped, f"Scale Start/End: {PUPIL_SCALE_START}/{PUPIL_SCALE_END}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_cropped, f"S1 Norm. Rad: {MIN_NORMALIZED_RADIUS}-{MAX_NORMALIZED_RADIUS}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # <-- Texto actualizado
        cv2.putText(frame_cropped, f"S2 Robust: {ROBUST_FILTER_THRESHOLD}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_cropped, f"S3 Morph: {MORPH_CLEANUP_KERNEL_SIZE}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_cropped, f"S4 Smooth Alpha: {IRIS_SMOOTHING_ALPHA}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame_cropped, f"Cleaned Points: {len(cleaned_points)}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame_cropped, "Pupila (Amarillo)", (10, h_frame - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame_cropped, "Límite Búsqueda (Naranja)", (10, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

        # Añadir textos a la vista del mapa (¡ACTUALIZADO!)
        text_offset_x = 10
        cv2.putText(final_bgr_display, "Mapa de Puntos", (text_offset_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(final_bgr_display, "Inliers (Blanco)", (text_offset_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(final_bgr_display, "Outliers S2 (Amarillo)", (text_offset_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(final_bgr_display, "Outliers S1 (Rojo)", (text_offset_x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(final_bgr_display, "Iris Fit (Magenta)", (text_offset_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        combined_view = np.hstack((frame_cropped, final_bgr_display))

        cv2.imshow(window_name, combined_view)

        # --- Keyboard Control ---
        processing_time = time.time() - start_time
        wait_time = max(1, int(frame_delay - (processing_time * 1000)))
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'): print("Processing stopped by user."); break
        elif key == ord(' '): cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# --- ENTRY POINT (Sin cambios) ---
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH): print(f"Error: Video file not found at: {VIDEO_PATH}")
    else: debug_full_frame_processing(VIDEO_PATH)