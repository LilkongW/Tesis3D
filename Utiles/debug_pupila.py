import cv2
import numpy as np
import os
import time
import math # <-- Añadido para la optimización de ángulos

# --- PARÁMETROS DE PREPROCESAMIENTO ---
GAUSSIAN_KERNEL_SIZE = (7, 7)
INITIAL_THRESHOLD_VALUE = 86   # Valor inicial para el trackbar de Umbral
INITIAL_CLAHE_CLIP_LIMIT = 2   # Valor inicial para el trackbar de CLAHE

# --- PARÁMETROS DE FILTRADO DE CONTORNOS (De tu script original) ---
MIN_PUPIL_AREA = 1200
MAX_PUPIL_AREA = 10000
MAX_ELLIPSE_RATIO = 2.8

# --- FUNCIONES DE UTILIDAD (Solo las necesarias) ---

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
    """Aplica umbral binario INVERSO: píxeles MÁS OSCUROS que el umbral se vuelven 255 (blanco)."""
    _, thresholded_image = cv2.threshold(image, int(threshold_value), 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# --- FUNCIÓN DE OPTIMIZACIÓN DE ÁNGULO (AÑADIDA) ---
def optimize_contours_by_angle(contours):
    """Filtra los puntos de un contorno basándose en el ángulo y la convexidad."""
    if len(contours) < 1 or len(contours[0]) < 5:
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
                
            # Calcular ángulo entre vectores
            dot_product = np.dot(vec1, vec2)
            angle = np.arccos(dot_product / (norm_vec1 * norm_vec2))

        # Vector al centroide
        vec_to_centroid = centroid - current_point
        
        # Filtro de convexidad (simplificado): el ángulo entre los vectores 
        # y el vector al centroide debe ser agudo.
        if np.dot(vec_to_centroid, (vec1+vec2)) > 0: # Comprueba si apunta 'hacia adentro'
             filtered_points.append(all_contours[i])
    
    if not filtered_points:
        return np.array([], dtype=np.int32).reshape((-1, 1, 2))

    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))


# Función vacía para el callback del trackbar
def on_trackbar(val):
    """Función placeholder requerida por createTrackbar."""
    pass

# --- FUNCIÓN PRINCIPAL DE DEBUG ---
def debug_binarization(video_path):
    """Procesa un video y muestra el original vs. el binarizado con un trackbar."""
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    frame_delay = int(1000 / fps)
    
    print(f"Procesando video: {video_path}")
    print("Mueve los trackbars para ajustar CLAHE y el Umbral.")
    print("Presiona 'q' para salir, 'espacio' para pausar.")

    # --- Crear ventana y trackbars ---
    window_name = "Debug Final (Elipse Optimizada | Binarizado)"
    cv2.namedWindow(window_name)
    
    # Trackbar para el Umbral
    cv2.createTrackbar("Threshold", window_name, INITIAL_THRESHOLD_VALUE, 255, on_trackbar)
    # Trackbar para el Clip Limit de CLAHE
    cv2.createTrackbar("Clip Limit", window_name, INITIAL_CLAHE_CLIP_LIMIT, 10, on_trackbar)
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        
        if not ret:
            print(f"Video terminado: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reiniciar el video
            continue

        # --- 1. Preprocesamiento ---
        frame_cropped = crop_to_aspect_ratio(frame)
        frame_blurred = cv2.GaussianBlur(frame_cropped, GAUSSIAN_KERNEL_SIZE, 0)
        gray_frame_original = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
        
        
        # --- 2. APLICAR CLAHE ---
        current_clip_limit = max(1, cv2.getTrackbarPos("Clip Limit", window_name))
        clahe = cv2.createCLAHE(clipLimit=float(current_clip_limit), tileGridSize=(8, 8))
        gray_frame_clahe = clahe.apply(gray_frame_original)

        
        # --- 3. Binarización ---
        current_threshold = cv2.getTrackbarPos("Threshold", window_name)
        thresholded_image = apply_fixed_binary_threshold(gray_frame_clahe, current_threshold)
        
        
        # --- 4. SELECCIÓN DEL CONTORNO MÁS OSCURO ---
        contours, _ = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_pupil_contour = None
        min_average_darkness = 256
        
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            
            # Filtros 1, 2 y 3 (Area, Puntos, Forma)
            if not (MIN_PUPIL_AREA <= contour_area <= MAX_PUPIL_AREA) or len(contour) < 5:
                continue

            try:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis = max(axes); minor_axis = min(axes)
                if minor_axis == 0: continue
                aspect_ratio = major_axis / minor_axis
                
                if aspect_ratio <= MAX_ELLIPSE_RATIO:
                    
                    # Filtro de Oscuridad (Medido en imagen CLAHE)
                    mask = np.zeros_like(gray_frame_clahe, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    dark_pixels = gray_frame_clahe[mask == 255]
                    
                    if dark_pixels.size == 0: continue
                    average_darkness = np.mean(dark_pixels)
                    
                    if average_darkness < min_average_darkness:
                        min_average_darkness = average_darkness
                        best_pupil_contour = contour
                        
            except Exception:
                continue 

        # --- 5. OPTIMIZACIÓN Y VISUALIZACIÓN ---
        
        if best_pupil_contour is not None:
            # --- Buscar Píxel más Oscuro (NUEVO) ---
            # Crear máscara solo para el mejor contorno
            mask = np.zeros_like(gray_frame_clahe, dtype=np.uint8)
            cv2.drawContours(mask, [best_pupil_contour], -1, 255, -1)
            
            # Encontrar el píxel más oscuro (valor mínimo) DENTRO de la máscara
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray_frame_clahe, mask=mask)
            
            # Dibujar un círculo en el píxel más oscuro
            cv2.circle(frame_cropped, minLoc, 5, (0, 0, 255), -1) # Círculo rojo

            # --- Optimizar contorno por ángulo (NUEVO) ---
            optimized_contour = optimize_contours_by_angle([best_pupil_contour])
            
            final_ellipse = None
            if len(optimized_contour) >= 5:
                final_ellipse = cv2.fitEllipse(optimized_contour)
            else:
                # Fallback si la optimización elimina demasiados puntos
                final_ellipse = cv2.fitEllipse(best_pupil_contour)
            
            # Dibujar la elipse final
            if final_ellipse:
                cv2.ellipse(frame_cropped, final_ellipse, (0, 255, 255), 2) # Elipse amarilla

        # --- Visualización de la ventana de debug ---
        thresholded_bgr = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
        combined_view = np.hstack((frame_cropped, thresholded_bgr))
        
        cv2.putText(combined_view, f"Threshold: {current_threshold}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_view, f"Clip Limit: {current_clip_limit}.0", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, combined_view)

        # --- Control de teclado ---
        processing_time = time.time() - start_time
        wait_time = max(1, int(frame_delay - (processing_time * 1000)))
        
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            print("Procesamiento detenido por el usuario.")
            break
        elif key == ord(' '):
            cv2.waitKey(0) # Pausa

    cap.release()
    cv2.destroyAllWindows()

# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    
    # CONFIGURACIÓN: Pon aquí la ruta directa de tu video
    VIDEO_PATH = r"C:\Users\Victor\Documents\Tesis2\Videos\Experimento_1\Victoria\ROI_videos_640x480\grabacion_experimento_ESP32CAM_5_ROI_640x480.mp4" # <-- CAMBIA ESTA RUTA
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: El archivo de video no existe en: {VIDEO_PATH}")
    else:
        debug_binarization(VIDEO_PATH)