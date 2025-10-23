import cv2
import numpy as np
import os
import time
import math # Añadido de nuevo para la optimización de ángulos

# --- PARÁMETROS DE PREPROCESAMIENTO ---
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0 # Fijo
INITIAL_THRESHOLD_VALUE = 70
INITIAL_KERNEL_N = 2 # N=2 -> (2*2)+1 = 5. Usaremos un kernel de 5x5

# --- PARÁMETROS DE FILTRADO (NUEVO) ---
MIN_PUPIL_AREA = 1000
MAX_PUPIL_AREA = 12000

# --- PARÁMETROS DE TRACKING (NUEVO) ---
ROI_SEARCH_DIM = 120 # Tamaño (ancho y alto) del ROI de búsqueda, ej. 120x120

# --- FUNCIONES DE UTILIDAD ---

def crop_to_aspect_ratio(image, width=640, height=480):
    """Recorta y reescala la imagen a la relación de aspecto deseada."""
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
    """Aplica umbral binario INVERSO."""
    _, thresholded_image = cv2.threshold(image, int(threshold_value), 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# --- FUNCIÓN DE PUNTO MÁS OSCURO (AÑADIDA) ---
def find_darkest_2x2(image):
    """
    Encuentra el bloque de 2x2 píxeles más oscuro en la imagen.
    Devuelve la coordenada (x, y) de la esquina superior izquierda de ese bloque.
    """
    min_sum = 1021 
    min_loc = (0, 0) # (x, y)
    H, W = image.shape
    
    # Asegurarse de que la imagen no esté vacía
    if H < 2 or W < 2:
        return (0, 0)
        
    for y in range(H - 1):
        for x in range(W - 1):
            s = int(image[y, x])     + int(image[y+1, x]) + \
                int(image[y, x+1])   + int(image[y+1, x+1])
            
            if s < min_sum:
                min_sum = s
                min_loc = (x, y)
                
    return min_loc

# --- FUNCIÓN DE OPTIMIZACIÓN DE ÁNGULO (AÑADIDA) ---
def optimize_contours_by_angle(contours):
    """Filtra los puntos de un contorno basándose en el ángulo y la convexidad."""
    if not isinstance(contours, list) or len(contours) < 1 or len(contours[0]) < 5:
        return np.array([], dtype=np.int32).reshape((-1, 1, 2))

    if len(contours[0].shape) == 2:
        all_contours = contours[0].reshape((-1, 1, 2))
    else:
        all_contours = contours[0]

    spacing = max(1, int(len(all_contours)/25))
    filtered_points = []
    
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
            dot_product = np.clip(dot_product / (norm_vec1 * norm_vec2), -1.0, 1.0)
            angle = np.arccos(dot_product)

        vec_to_centroid = centroid - current_point
        
        if np.dot(vec_to_centroid, (vec1+vec2)) > 0: 
            filtered_points.append(all_contours[i])
    
    if not filtered_points or len(filtered_points) < 5:
        return np.array([], dtype=np.int32).reshape((-1, 1, 2))

    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))


# Función vacía para el callback del trackbar
def on_trackbar(val):
    """Función placeholder requerida por createTrackbar."""
    pass

# --- FUNCIÓN PRINCIPAL DE DEBUG (Paso 5 con ROI) ---
def debug_step_5_with_roi(video_path):
    """Combina todo: Búsqueda de punto oscuro + Filtro + Optimización + ROI."""
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    frame_delay = int(1000 / fps)
    
    print(f"Procesando video (Paso 5 - Final con ROI): {video_path}")
    print("Mueve los trackbars de 'Threshold' y 'Kernel Size'.")
    print("Presiona 'q' para salir, 'espacio' para pausar.")

    # --- Crear ventana y trackbars ---
    window_name = "Debug Final con Tracking (ROI) | Mascara Limpia"
    cv2.namedWindow(window_name)
    
    cv2.createTrackbar("Threshold", window_name, INITIAL_THRESHOLD_VALUE, 255, on_trackbar)
    cv2.createTrackbar("Kernel Size (N)", window_name, INITIAL_KERNEL_N, 5, on_trackbar)
    
    # --- Crear objeto CLAHE (una sola vez) ---
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    
    # --- VARIABLES DE ESTADO DE TRACKING (NUEVO) ---
    last_known_center = None
    tracking_active = False # Empezamos en modo Detección
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        
        if not ret:
            print(f"Video terminado: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reiniciar el video
            last_known_center = None # Reiniciar tracking
            tracking_active = False
            continue

        # --- 1. Preprocesamiento (Completo) ---
        frame_cropped = crop_to_aspect_ratio(frame)
        frame_blurred = cv2.GaussianBlur(frame_cropped, GAUSSIAN_KERNEL_SIZE, 0)
        gray_frame_original = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
        
        H, W = frame_cropped.shape[:2] # Obtenemos dimensiones 640x480
        
        # --- 2. APLICAR CLAHE (Completo) ---
        gray_frame_clahe = clahe.apply(gray_frame_original)
        
        # --- LÓGICA DE ROI vs. DETECCIÓN COMPLETA (NUEVO) ---
        roi_coords = (0, 0, W, H) 
        offset = (0, 0)
        
        if tracking_active and last_known_center is not None:
            # --- Modo Tracking: Definir ROI ---
            cx, cy = last_known_center
            half_dim = ROI_SEARCH_DIM // 2
            
            x1 = max(0, cx - half_dim)
            y1 = max(0, cy - half_dim)
            x2 = min(W, cx + half_dim)
            y2 = min(H, cy + half_dim)
            
            roi_coords = (x1, y1, x2, y2)
            offset = (x1, y1)
            
            # Recortar la imagen CLAHE al ROI
            roi_gray_clahe = gray_frame_clahe[y1:y2, x1:x2]
        
        else:
            # --- Modo Detección: Usar imagen completa ---
            roi_gray_clahe = gray_frame_clahe
        
        # Asegurarse de que el ROI no esté vacío
        if roi_gray_clahe.size == 0:
            roi_gray_clahe = gray_frame_clahe
            roi_coords = (0, 0, W, H)
            offset = (0, 0)
            tracking_active = False
            
        # --- 3. ENCONTRAR PUNTO ANCLA (EL MÁS OSCURO) ---
        dark_point_local = find_darkest_2x2(roi_gray_clahe) # (x, y) local
        dark_point = (dark_point_local[0] + offset[0], dark_point_local[1] + offset[1]) # (x, y) global
        
        # --- 4. BINARIZACIÓN Y MORFOLOGÍA (Solo en el ROI) ---
        current_threshold = cv2.getTrackbarPos("Threshold", window_name)
        current_kernel_n = max(1, cv2.getTrackbarPos("Kernel Size (N)", window_name))
        
        kernel_size = (current_kernel_n * 2) + 1
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        thresholded_image_raw = apply_fixed_binary_threshold(roi_gray_clahe, current_threshold)
        thresholded_image_closed = cv2.morphologyEx(thresholded_image_raw, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        thresholded_image_final = cv2.morphologyEx(thresholded_image_closed, cv2.MORPH_OPEN, morph_kernel, iterations=1)

        # --- 5. FILTRADO DE CONTORNO POR PUNTO ANCLA (Solo en el ROI) ---
        contours, _ = cv2.findContours(thresholded_image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_pupil_contour_local = None
        best_pupil_contour_global = None
        
        for contour_local in contours:
            
            # --- FILTRO DE ÁREA ---
            contour_area = cv2.contourArea(contour_local)
            if not (MIN_PUPIL_AREA <= contour_area <= MAX_PUPIL_AREA):
                continue 

            # Filtro de punto ancla (local vs local)
            if cv2.pointPolygonTest(contour_local, dark_point_local, False) >= 0:
                best_pupil_contour_local = contour_local
                best_pupil_contour_global = contour_local + (offset[0], offset[1]) # Guardar global
                break 

        # Preparar la visualización de la derecha (máscara)
        final_bgr = cv2.cvtColor(thresholded_image_final, cv2.COLOR_GRAY2BGR) # Máscara (pequeña)

        # --- 6. OPTIMIZACIÓN Y DIBUJO ---
        if best_pupil_contour_global is not None:
            # Dibujar el contorno seleccionado en la máscara (verde)
            cv2.drawContours(final_bgr, [best_pupil_contour_local], -1, (0, 255, 0), 1)
            
            # Aplicar la optimización de ángulo al contorno GLOBAL
            optimized_contour = optimize_contours_by_angle([best_pupil_contour_global])
            
            try:
                final_ellipse = None
                if len(optimized_contour) >= 5:
                    final_ellipse = cv2.fitEllipse(optimized_contour)
                else:
                    final_ellipse = cv2.fitEllipse(best_pupil_contour_global)
                
                if final_ellipse:
                    cv2.ellipse(frame_cropped, final_ellipse, (0, 255, 255), 2)
                    
                    # --- ACTUALIZAR ESTADO DE TRACKING ---
                    tracking_active = True
                    last_known_center = (int(final_ellipse[0][0]), int(final_ellipse[0][1]))
            
            except cv2.error:
                tracking_active = False # Falló el ajuste
                last_known_center = None
        else:
            # --- Pupila NO encontrada ---
            tracking_active = False
            last_known_center = None
        
        # Dibujar el punto ancla GLOBAL (rojo) en la imagen original
        cv2.circle(frame_cropped, dark_point, 5, (0, 0, 255), -1)
        
        # Dibujar el ROI si está activo
        if tracking_active:
            x1, y1, x2, y2 = roi_coords
            cv2.rectangle(frame_cropped, (x1, y1), (x2, y2), (255, 0, 0), 2) # Rectángulo azul
            
        # --- 7. VISUALIZACIÓN FINAL ---
        
        # Preparar vista combinada (manejando el tamaño del ROI)
        display_mask = np.zeros_like(frame_cropped) # Fondo negro 640x480
        x1, y1, x2, y2 = roi_coords
        
        try:
            # Pegar el ROI procesado (final_bgr) en el fondo negro
            display_mask[y1:y2, x1:x2] = final_bgr
        except ValueError as e:
            # Fallback por si las formas no coinciden
            h_roi, w_roi, _ = final_bgr.shape
            display_mask[0:h_roi, 0:w_roi] = final_bgr
            
        combined_view = np.hstack((frame_cropped, display_mask))
        
        cv2.putText(combined_view, f"Threshold: {current_threshold}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_view, f"Kernel Size: {kernel_size}x{kernel_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
    VIDEO_PATH = r"C:\Users\Victor\Documents\Tesis3D\Videos\Experimento_1\Victor\ROI_videos_640x480\grabacion_experimento_ESP32CAM_1_ROI_640x480.mp4" # <-- CAMBIA ESTA RUTA
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: El archivo de video no existe en: {VIDEO_PATH}")
    else:
        debug_step_5_with_roi(VIDEO_PATH) # Renombrada la función