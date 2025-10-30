import cv2
import numpy as np
import os
import time
import math 

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

# --- <<<--- PARÁMETROS IRIS (Tus ajustes) ---
IRIS_MAX_SEARCH_SCALE_FACTOR = 4.0 
IRIS_RADIAL_RAYS = 400
IRIS_GRADIENT_THRESHOLD = 0
PUPIL_RADIUS_BUFFER_RATIO = 1.4
IRIS_SPECULAR_LOOKAHEAD_PIXELS = 25
SPECULAR_GRADIENT_RATIO = -0.6 
# --- <<<--- ---

# --- <<<--- PARÁMETRO DE FILTRO ROBUSTO ---
ROBUST_FILTER_THRESHOLD = 1.2
# --- <<<--- ---

# --- <<<--- PARÁMETRO DE SUAVIZADO ---
IRIS_SMOOTHING_ALPHA = 0.7
# --- <<<--- ---

VIDEO_PATH = r"/home/vit/Documentos/Tesis3D/Videos/Experimento_2/Victor/Victor_intento_1.mp4" # <-- CHANGE THIS PATH

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

    window_name = "Debug Full Frame | Iris Detection"
    cv2.namedWindow(window_name)
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

    # --- Inicializar el radio suavizado y la elipse suavizada ---
    smoothed_iris_radius = 0.0
    # Inicializamos con valores que fitEllipse no produciría, para el primer frame
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
        cv2.drawContours(final_bgr_display, discarded_horizontal_contours, -1, (0, 0, 255), 1) # Red

        if best_pupil_contour is not None:
            cv2.drawContours(final_bgr_display, [best_pupil_contour], -1, (0, 255, 0), 2) # Green
            
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
        # --- INICIO: LÓGICA DE DETECCIÓN DE IRIS (RADIAL) ---
        # --- ############################################# ---
        
        iris_edge_points = []
        current_fitted_iris_ellipse = None # Para almacenar la elipse ajustada en este frame
        
        if final_ellipse:
            
            # --- 1. Dibujar la pupila (en el frame original) ---
            cv2.ellipse(frame_cropped, final_ellipse, (0, 255, 255), 2) # Yellow
            
            # --- 2. Obtener parámetros de la pupila ---
            (cx_f, cy_f), (pupil_w_axis, pupil_h_axis), angle = final_ellipse
            cx, cy = int(cx_f), int(cy_f)
            pupil_radius = pupil_diameter / 2.0
            
            min_search_radius = int(pupil_radius * PUPIL_RADIUS_BUFFER_RATIO)
            max_search_radius = int(pupil_radius * IRIS_MAX_SEARCH_SCALE_FACTOR)

            # Dibujar zonas de búsqueda
            cv2.circle(frame_cropped, (cx, cy), min_search_radius, (0, 255, 0), 1) # Verde
            cv2.circle(frame_cropped, (cx, cy), max_search_radius, (0, 165, 255), 1) # Naranja
            
            # --- 3. Lanzar Rayos ---
            for i in range(IRIS_RADIAL_RAYS):
                current_angle = (i / IRIS_RADIAL_RAYS) * 2 * np.pi
                cos_a = np.cos(current_angle)
                sin_a = np.sin(current_angle)
                
                candidate_gradients = []
                
                x_prev = int(cx + min_search_radius * cos_a)
                y_prev = int(cy + min_search_radius * sin_a)
                
                if not (0 <= x_prev < w_frame and 0 <= y_prev < h_frame):
                    continue
                prev_intensity = int(gray_frame_clahe[y_prev, x_prev])

                # --- 4. Caminar a lo largo del rayo ---
                for r in range(min_search_radius + 1, max_search_radius):
                    x_curr = int(cx + r * cos_a)
                    y_curr = int(cy + r * sin_a)
                    
                    if not (0 <= x_curr < w_frame and 0 <= y_curr < h_frame):
                        break 
                    
                    curr_intensity = int(gray_frame_clahe[y_curr, x_curr])
                    gradient = curr_intensity - prev_intensity
                    
                    # (Tus parámetros: IRIS_GRADIENT_THRESHOLD = 0)
                    if gradient > IRIS_GRADIENT_THRESHOLD:
                        candidate_gradients.append( (gradient, x_curr, y_curr, r) )
                        
                    prev_intensity = curr_intensity
                
                # --- 6. Validar candidatos ---
                if not candidate_gradients:
                    continue
                
                validated_candidates = []
                
                for candidate in candidate_gradients:
                    grad_val, x_c, y_c, r_c = candidate
                    is_reflection = False
                    
                    try:
                        intensity_at_peak = int(gray_frame_clahe[y_c, x_c])
                    except IndexError:
                        continue 
                    
                    for r_ahead in range(1, IRIS_SPECULAR_LOOKAHEAD_PIXELS + 1):
                        x_ahead = int(cx + (r_c + r_ahead) * cos_a)
                        y_ahead = int(cy + (r_c + r_ahead) * sin_a)
                        
                        if not (0 <= x_ahead < w_frame and 0 <= y_ahead < h_frame):
                            break
                        
                        intensity_after = int(gray_frame_clahe[y_ahead, x_ahead])
                        negative_gradient = intensity_after - intensity_at_peak
                        
                        if negative_gradient < (IRIS_GRADIENT_THRESHOLD * SPECULAR_GRADIENT_RATIO):
                            is_reflection = True
                            break
                    
                    if not is_reflection:
                        # Guardar el punto validado (x, y) y su radio (r_c)
                        validated_candidates.append( ((x_c, y_c), r_c) )
                
                # --- 7. Seleccionar el punto VÁLIDO con el MENOR RADIO ---
                if validated_candidates:
                    # Ordenar por radio (r_c), de MENOR a mayor
                    validated_candidates.sort(key=lambda x: x[1], reverse=False)
                    # Quedarse con el punto del candidato con menor radio
                    best_point = validated_candidates[0][0] 
                    iris_edge_points.append(best_point)

            # --- 10. FILTRAR PUNTOS Y AJUSTAR ELIPSE ---
            
            if len(iris_edge_points) >= 10:
                points_np = np.array(iris_edge_points)
                
                # --- LÓGICA DE FILTRADO DE OUTLIERS ---
                center_np = np.array([cx, cy])
                radii = np.linalg.norm(points_np - center_np, axis=1)
                
                median_radius = np.median(radii)
                mad = np.median(np.abs(radii - median_radius)) * 1.4826 
                
                if mad == 0: mad = 1.0 
                
                z_score_robusta = np.abs(radii - median_radius) / mad 
                inlier_indices = np.where(z_score_robusta < ROBUST_FILTER_THRESHOLD)
                cleaned_points_np = points_np[inlier_indices]
                
                # --- Dibujar inliers y outliers para debug ---
                for pt in cleaned_points_np:
                     cv2.circle(final_bgr_display, pt, 2, (255, 255, 255), -1) # Inliers (Blanco)
                
                outlier_indices = np.where(z_score_robusta >= ROBUST_FILTER_THRESHOLD)
                outlier_points_np = points_np[outlier_indices]
                for pt in outlier_points_np:
                    cv2.circle(final_bgr_display, pt, 2, (0, 0, 255), -1) # Outliers (Rojo)

                # --- ¡AJUSTE DE ELIPSE AL IRIS! ---
                if len(cleaned_points_np) >= 5: # fitEllipse requiere al menos 5 puntos
                    try:
                        fitted_iris_ellipse = cv2.fitEllipse(cleaned_points_np)
                        current_fitted_iris_ellipse = fitted_iris_ellipse

                        # Para el suavizado, usaremos el radio promedio de la elipse actual
                        (center_e, axes_e, angle_e) = fitted_iris_ellipse
                        current_iris_radius = (axes_e[0] + axes_e[1]) / 4.0 # Diametros -> radios -> promedio
                        
                        # --- ¡CLIPPING! ---
                        current_iris_radius = min(current_iris_radius, max_search_radius)

                        # --- ¡SUAVIZADO (EMA)! ---
                        # Suavizamos el radio para el texto, y si decidimos, la elipse completa
                        if smoothed_iris_radius == 0.0: # Primera detección
                            smoothed_iris_radius = current_iris_radius
                            smoothed_iris_ellipse = fitted_iris_ellipse
                        else:
                            smoothed_iris_radius = (current_iris_radius * IRIS_SMOOTHING_ALPHA) + (smoothed_iris_radius * (1.0 - IRIS_SMOOTHING_ALPHA))
                            # Esto es un suavizado simple de la elipse, se puede mejorar
                            # Suavizar el centro (x,y)
                            smoothed_center_x = (fitted_iris_ellipse[0][0] * IRIS_SMOOTHING_ALPHA) + (smoothed_iris_ellipse[0][0] * (1.0 - IRIS_SMOOTHING_ALPHA))
                            smoothed_center_y = (fitted_iris_ellipse[0][1] * IRIS_SMOOTHING_ALPHA) + (smoothed_iris_ellipse[0][1] * (1.0 - IRIS_SMOOTHING_ALPHA))
                            # Suavizar los ejes (ancho, alto)
                            smoothed_axes_w = (fitted_iris_ellipse[1][0] * IRIS_SMOOTHING_ALPHA) + (smoothed_iris_ellipse[1][0] * (1.0 - IRIS_SMOOTHING_ALPHA))
                            smoothed_axes_h = (fitted_iris_ellipse[1][1] * IRIS_SMOOTHING_ALPHA) + (smoothed_iris_ellipse[1][1] * (1.0 - IRIS_SMOOTHING_ALPHA))
                            # Suavizar el ángulo
                            smoothed_angle = (fitted_iris_ellipse[2] * IRIS_SMOOTHING_ALPHA) + (smoothed_iris_ellipse[2] * (1.0 - IRIS_SMOOTHING_ALPHA))

                            smoothed_iris_ellipse = ((smoothed_center_x, smoothed_center_y), (smoothed_axes_w, smoothed_axes_h), smoothed_angle)

                    except cv2.error:
                        # Si falla el ajuste, reiniciamos el radio y la elipse para suavizado
                        smoothed_iris_radius = 0.0
                        smoothed_iris_ellipse = ((0,0),(0,0),0) 

            # Dibujar la ELIPSE SUAVIZADA (o la ajustada si no hay historial)
            if smoothed_iris_ellipse[1][0] > 0 and smoothed_iris_ellipse[1][1] > 0: # Solo si los ejes son válidos
                cv2.ellipse(frame_cropped, smoothed_iris_ellipse, (255, 0, 255), 2) # Magenta
                cv2.ellipse(final_bgr_display, smoothed_iris_ellipse, (255, 0, 255), 2) # Magenta


        # --- ############################################# ---
        # --- FIN: LÓGICA DE DETECCIÓN DE IRIS ---
        # --- ############################################# ---

        # --- 7. Final Visualization ---
        combined_view = np.hstack((frame_cropped, final_bgr_display))
        
        # Textos de Info (Izquierda)
        cv2.putText(combined_view, f"Pupil Thresh: {FIXED_THRESHOLD_VALUE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_view, f"Pupil Diameter: {pupil_diameter:.1f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(combined_view, f"Iris Rays: {IRIS_RADIAL_RAYS}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(combined_view, f"Iris Grad Thresh: {IRIS_GRADIENT_THRESHOLD}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(combined_view, f"Iris Points: {len(iris_edge_points)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(combined_view, f"Iris Smooth Alpha: {IRIS_SMOOTHING_ALPHA}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(combined_view, f"Final Iris Radius (avg): {smoothed_iris_radius:.1f}px", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        
        # Textos de leyenda en la máscara (Derecha) - ACTUALIZADO
        text_offset_x = 650
        cv2.putText(combined_view, "Pupil Area OK (Azul)", (text_offset_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        cv2.putText(combined_view, "Pupil Good Fit (Cyan)", (text_offset_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(combined_view, "Pupil Too Horizontal (Rojo)", (text_offset_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(combined_view, "Final Pupil (Verde)", (text_offset_x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(combined_view, "Inliers (Blanco)", (text_offset_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(combined_view, "Outliers (Rojo)", (text_offset_x, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(combined_view, "Final Iris (Magenta)", (text_offset_x, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        # Textos de leyenda en el frame (Izquierda, abajo)
        cv2.putText(combined_view, "Iris (Magenta)", (10, h_frame - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        cv2.putText(combined_view, "Pupila (Amarillo)", (10, h_frame - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(combined_view, "Inicio Búsqueda (Verde)", (10, h_frame - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(combined_view, "Fin Búsqueda (Naranja)", (10, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)


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