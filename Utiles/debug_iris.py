import cv2
import numpy as np
import os
import time
import math 
# (matplotlib e io eliminados, ya no son necesarios)

# --- PARÁMETROS DE PREPROCESAMIENTO (FIJOS) ---
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0 
FIXED_THRESHOLD_VALUE = 30 
FIXED_KERNEL_N = 2         

# --- PARÁMETROS DE FILTRADO PUPILA (FIJOS) ---
FIXED_MIN_PUPIL_AREA = 1000 
FIXED_MAX_PUPIL_AREA = 8000  

# --- PARÁMETRO DE ROBUSTEZ (FILTRO DE FIT ELÍPTICO) ---
MIN_ELLIPTICAL_FIT_RATIO = 0.85  
MAX_ELLIPTICAL_FIT_RATIO = 1.20 
# ----------------------------------------------

# --- PARÁMETRO DE FILTRO DE BBOX PUPILA (FIJO) ---
HORIZONTALITY_TOLERANCE = 1.30 
# -----------------------------------------------

# --- <<<--- PARÁMETROS DEL MAPA DE GRADIENTES ---
IRIS_MAX_SEARCH_SCALE_FACTOR = 5
IRIS_RADIAL_RAYS = 400
POSITIVE_GRADIENT_THRESHOLD = 3 # Umbral para 'puntos verdes'
RING_END_THRESHOLD = -2 # Umbral negativo para detectar el FIN de un anillo
# --- <<<--- ---

# --- FILTRO ETAPA 1 (ESTÁTICO) ---
MIN_RADIUS_THRESHOLD = 55.0
MAX_RADIUS_THRESHOLD = 110.0
# --- <<<--- ---

# --- FILTRO ETAPA 2 (DINÁMICO) ---
ROBUST_FILTER_THRESHOLD = 2 # Umbral de pertenencia (Z-score robusto)
# --- <<<--- ---

# --- ETAPA 3 (MORFOLÓGICO) ---
MORPH_CLEANUP_KERNEL_SIZE = 5 # Tamaño del kernel para rellenar huecos
# --- <<<--- ---

# ATENCIÓN: Confirma que esta ruta es correcta
VIDEO_PATH = r"/home/vit/Documentos/Tesis3D/Videos/Experimento_3/Raul/Raul_intento_1.mp4" 

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

# --- FUNCIÓN DE GRÁFICO DE CAJAS (ELIMINADA) ---

# --- FUNCIÓN PRINCIPAL DE DEBUG (Modificada) ---
def debug_full_frame_processing(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video: {video_path}"); return
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30
    frame_delay = int(1000 / fps)

    window_name = "Debug Iris Fit" # <-- Título actualizado
    cv2.namedWindow(window_name)
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        # --- 1. Preprocessing ---
        frame_cropped = crop_to_aspect_ratio(frame)
        h_frame, w_frame = frame_cropped.shape[:2] 
        
        frame_blurred = cv2.GaussianBlur(frame_cropped, GAUSSIAN_KERNEL_SIZE, 0)
        gray_frame_original = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)

        # --- 2. CLAHE ---
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
        cv2.drawContours(final_bgr_display, contours_in_area_range, -1, (255, 0, 0), 1) # Blue
        cv2.drawContours(final_bgr_display, good_fit_contours, -1, (255, 255, 0), 1) # Cyan
        cv2.drawContours(final_bgr_display, discarded_horizontal_contours, -1, (0, 0, 100), 1) # Red (oscuro)

        if best_pupil_contour is not None:
            cv2.drawContours(final_bgr_display, [best_pupil_contour], -1, (0, 100, 0), 2) # Green (oscuro)
            
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
        # --- INICIO: LÓGICA DE MAPEO DE GRADIENTES ---
        # --- ############################################# ---
        
        inner_texture_points = []
        radii = [] 
        final_inlier_points = [] 
        fitted_iris_ellipse = None # <-- ¡NUEVO!
        
        if final_ellipse:
            
            (cx_f, cy_f), (pupil_w_axis, pupil_h_axis), angle = final_ellipse
            cx, cy = int(cx_f), int(cy_f)
            pupil_radius = pupil_diameter / 2.0
            
            min_search_radius = 0 
            max_search_radius = int(pupil_radius * IRIS_MAX_SEARCH_SCALE_FACTOR)
            
            # --- 3. Lanzar Rayos ---
            for i in range(IRIS_RADIAL_RAYS):
                current_angle = (i / IRIS_RADIAL_RAYS) * 2 * np.pi
                cos_a = np.cos(current_angle)
                sin_a = np.sin(current_angle)
                
                x_prev = cx
                y_prev = cy
                
                if not (0 <= x_prev < w_frame and 0 <= y_prev < h_frame):
                    continue
                prev_intensity = int(gray_frame_clahe[y_prev, x_prev])

                state = "SEARCHING"

                # --- 4. Caminar a lo largo del rayo y PINTAR MAPA ---
                for r in range(1, max_search_radius):
                    x_curr = int(cx + r * cos_a)
                    y_curr = int(cy + r * sin_a)
                    
                    if not (0 <= x_curr < w_frame and 0 <= y_curr < h_frame):
                        break 
                    
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
                            try:
                                final_bgr_display[y_curr, x_curr] = (0, 255, 0) # Verde
                                inner_texture_points.append((x_curr, y_curr))
                            except IndexError: pass
                            state = "DONE"

                    elif state == "DONE":
                        break
                        
                    prev_intensity = curr_intensity

            # --- 5. FILTRO DE TRES ETAPAS ---
            if len(inner_texture_points) > 0:
                center_np = np.array([cx, cy])
                points_np = np.array(inner_texture_points)
                
                radii = np.linalg.norm(points_np - center_np, axis=1) # Radios de *todos* los puntos
                
                # --- ETAPA 1: Filtro Estático (Hardcoded) ---
                stage_one_inliers_pts = []
                stage_one_inliers_rad = []
                stage_one_outliers_pts = []
                
                for i, pt in enumerate(inner_texture_points):
                    r = radii[i]
                    if MIN_RADIUS_THRESHOLD <= r <= MAX_RADIUS_THRESHOLD:
                        stage_one_inliers_pts.append(pt)
                        stage_one_inliers_rad.append(r)
                    else:
                        stage_one_outliers_pts.append(pt)
                
                # --- ETAPA 2: Filtro Dinámico (Mediana) ---
                stage_two_inliers_pts = [] 
                stage_two_outliers_pts = [] 

                if len(stage_one_inliers_pts) >= 5: 
                    median_radius = np.median(stage_one_inliers_rad)
                    mad = np.median(np.abs(stage_one_inliers_rad - median_radius)) * 1.4826 
                    if mad == 0: mad = 1.0 
                    
                    for i, pt in enumerate(stage_one_inliers_pts):
                        r = stage_one_inliers_rad[i]
                        z_score_robusta = np.abs(r - median_radius) / mad
                        
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
                    final_inlier_points = list(zip(coords[1], coords[0])) # (x, y)

                # --- ¡NUEVO! ETAPA 4: AJUSTE DE ELIPSE ---
                if len(final_inlier_points) >= 5:
                    try:
                        # Convertir lista de tuplas a array de numpy
                        final_points_np = np.array(final_inlier_points, dtype=np.int32).reshape((-1, 1, 2))
                        fitted_iris_ellipse = cv2.fitEllipse(final_points_np)
                    except cv2.error as e:
                        print(f"Error en fitEllipse: {e}")
                        fitted_iris_ellipse = None
                
                # --- Dibujar los puntos clasificados (Actualizado) ---
                for pt in stage_one_outliers_pts:
                    cv2.circle(final_bgr_display, pt, 2, (0, 0, 255), -1) # Outliers Etapa 1 (Rojo)
                for pt in stage_two_outliers_pts:
                    cv2.circle(final_bgr_display, pt, 2, (0, 255, 255), -1) # Outliers Etapa 2 (Amarillo)
                for pt in final_inlier_points:
                    cv2.circle(final_bgr_display, pt, 2, (255, 255, 255), -1) # Inliers Finales (Blanco)

        # --- ############################################# ---
        # --- FIN: LÓGICA DE MAPEO DE GRADIENTES ---
        # --- ############################################# ---

        # --- 7. Final Visualization ---
        
        # Fila superior
        cv2.ellipse(frame_cropped, final_ellipse, (0, 255, 255), 2) # Pupila (Amarillo)
        cv2.circle(frame_cropped, (cx, cy), max_search_radius, (0, 165, 255), 1) # Límite (Naranja)
        
        # --- ¡NUEVO! Dibujar elipse de iris ajustada ---
        if fitted_iris_ellipse is not None:
            cv2.ellipse(frame_cropped, fitted_iris_ellipse, (255, 0, 255), 2) # Iris (Magenta)
            cv2.ellipse(final_bgr_display, fitted_iris_ellipse, (255, 0, 255), 2) # Iris (Magenta)

        # Añadir textos a la vista del frame
        cv2.putText(frame_cropped, f"Pupil Diameter: {pupil_diameter:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_cropped, f"Pos Grad (Verde): {POSITIVE_GRADIENT_THRESHOLD}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_cropped, f"Ring End Thresh: {RING_END_THRESHOLD}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_cropped, f"Min/Max Radius: {MIN_RADIUS_THRESHOLD}/{MAX_RADIUS_THRESHOLD}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_cropped, f"Robust Thresh (S2): {ROBUST_FILTER_THRESHOLD}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_cropped, f"Morph Kernel (S3): {MORPH_CLEANUP_KERNEL_SIZE}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_cropped, f"Final Points: {len(final_inlier_points)}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Añadir textos a la vista del mapa
        text_offset_x = 10
        cv2.putText(final_bgr_display, "Mapa de Puntos", (text_offset_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(final_bgr_display, "Inliers (Blanco)", (text_offset_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(final_bgr_display, "Outliers S2 (Amarillo)", (text_offset_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(final_bgr_display, "Outliers S1 (Rojo)", (text_offset_x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(final_bgr_display, "Iris Fit (Magenta)", (text_offset_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        combined_view = np.hstack((frame_cropped, final_bgr_display))

        # (Gráfico de cajas eliminado, 'vstack' ya no es necesario)

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