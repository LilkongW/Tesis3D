import cv2
import numpy as np
import os
import time
import math 

# --- PARÁMETROS DE PREPROCESAMIENTO ---
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0 
INITIAL_THRESHOLD_VALUE = 50
INITIAL_KERNEL_N = 2 

# --- PARÁMETROS DE FILTRADO (ELIMINADOS) ---
# --- PARÁMETRO DE ROBUSTEZ (ELIMINADO) ---
# --- PARÁMETRO DE FILTRO DE BBOX (ELIMINADO) ---

VIDEO_PATH = r"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/Arteaga/Arteaga_intento_5.mp4" # <-- CHANGE THIS PATH

# --- FUNCIONES DE UTILIDAD ---

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

# --- ¡NUEVO! FUNCIÓN DE OPTIMIZACIÓN RE-INCORPORADA ---
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
# --- --- ---

def on_trackbar(val): pass

# --- FUNCIÓN PRINCIPAL DE DEBUG (Modificada) ---
def debug_full_frame_processing(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video: {video_path}"); return
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30
    frame_delay = int(1000 / fps)

    print(f"Processing video (Darkest Seed Select): {video_path}")
    print("Adjust Threshold and Kernel Size sliders.")
    print("Press 'q' to quit, 'space' to pause.")

    window_name = "Debug | Darkest Seed (Red) -> Selects Contour (Green) -> Fit (Yellow)"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Threshold", window_name, INITIAL_THRESHOLD_VALUE, 255, on_trackbar)
    cv2.createTrackbar("Kernel Size (N)", window_name, INITIAL_KERNEL_N, 5, on_trackbar)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        # --- 1. Preprocessing ---
        frame_cropped = crop_to_aspect_ratio(frame)
        frame_blurred = cv2.GaussianBlur(frame_cropped, GAUSSIAN_KERNEL_SIZE, 0)
        
        # --- <<<--- ¡LÍNEA CORREGIDA! ---
        gray_frame_original = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
        # --- <<<--- ---

        # --- 2. CLAHE ---
        gray_frame_clahe = clahe.apply(gray_frame_original)

        # --- 3. Encontrar el bloque 5x5 más oscuro ---
        avg_intensity_map = cv2.blur(gray_frame_clahe, (5, 5))
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(avg_intensity_map)
        darkest_center_point = minLoc
        
        # --- 4. Binarization & Morphology ---
        current_threshold = cv2.getTrackbarPos("Threshold", window_name)
        current_kernel_n = max(1, cv2.getTrackbarPos("Kernel Size (N)", window_name))
        kernel_size = (current_kernel_n * 2) + 1
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        thresholded_image_raw = apply_fixed_binary_threshold(gray_frame_clahe, current_threshold)
        thresholded_image_closed = cv2.morphologyEx(thresholded_image_raw, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        thresholded_image_final = cv2.morphologyEx(thresholded_image_closed, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        
        # --- 5. Lógica de Selección de Contorno (¡NUEVO!) ---
        
        # Encontrar TODOS los contornos en la máscara
        contours, _ = cv2.findContours(thresholded_image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_pupil_contour = None
        
        if darkest_center_point:
            for contour in contours:
                # Comprobar si el punto más oscuro está DENTRO de este contorno
                if cv2.pointPolygonTest(contour, darkest_center_point, False) >= 0:
                    best_pupil_contour = contour
                    break # Encontramos nuestro contorno, salimos del bucle
        
        # --- 6. Optimización y Ajuste de Elipse ---
        
        final_ellipse = None
        if best_pupil_contour is not None:
            
            # Aplicar la optimización de ángulo al contorno seleccionado
            optimized_contour = optimize_contours_by_angle([best_pupil_contour])
            
            # Intentar ajustar la elipse al contorno optimizado
            try:
                if len(optimized_contour) >= 5:
                    final_ellipse = cv2.fitEllipse(optimized_contour)
                elif len(best_pupil_contour) >= 5: # Fallback al contorno original si la optimización falló
                    final_ellipse = cv2.fitEllipse(best_pupil_contour)
            except cv2.error:
                final_ellipse = None # El ajuste falló
        
        # --- 7. Final Visualization ---
        
        # Convertir la máscara final a BGR para poder dibujar colores
        final_bgr_mask = cv2.cvtColor(thresholded_image_final, cv2.COLOR_GRAY2BGR)
        
        # Dibujar TODOS los contornos en la máscara en AZUL
        cv2.drawContours(final_bgr_mask, contours, -1, (255, 0, 0), 1) # Blue
        
        # Dibujar la cruz de la "semilla" (ROJA)
        if darkest_center_point:
            cv2.line(frame_cropped, (darkest_center_point[0] - 10, darkest_center_point[1]), (darkest_center_point[0] + 10, darkest_center_point[1]), (0, 0, 255), 2)
            cv2.line(frame_cropped, (darkest_center_point[0], darkest_center_point[1] - 10), (darkest_center_point[0], darkest_center_point[1] + 10), (0, 0, 255), 2)

        # Si encontramos un contorno, dibujarlo en VERDE (en la máscara)
        if best_pupil_contour is not None:
            cv2.drawContours(final_bgr_mask, [best_pupil_contour], -1, (0, 255, 0), 2) # Green
        
        # Si logramos ajustar una elipse, dibujarla en AMARILLO (en el frame)
        if final_ellipse is not None:
            cv2.ellipse(frame_cropped, final_ellipse, (0, 255, 255), 2) # Yellow
            
        combined_view = np.hstack((frame_cropped, final_bgr_mask))
        cv2.putText(combined_view, f"Threshold: {current_threshold}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_view, f"Kernel Size: {kernel_size}x{kernel_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        h_frame = frame_cropped.shape[0]
        cv2.putText(combined_view, "Original + Seed (Red) + Fit (Yellow)", (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(combined_view, "Mask + All Contours (Blue) + Selected (Green)", (650, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)


        cv2.imshow(window_name, combined_view)

        # --- Keyboard Control ---
        processing_time = time.time() - start_time
        wait_time = max(1, int(frame_delay - (processing_time * 1000)))
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'): print("Processing stopped by user."); break
        elif key == ord(' '): cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# --- ENTRY POINT ---
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH): print(f"Error: Video file not found at: {VIDEO_PATH}")
    else: debug_full_frame_processing(VIDEO_PATH)