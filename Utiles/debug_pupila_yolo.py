import cv2
import numpy as np
import os
import time
import math 
from ultralytics import YOLO

# --- ⚠️ CONFIGURACIÓN DINÁMICA DE RUTAS ---

# 1. Definir la ruta base según el sistema operativo
if os.name == 'nt': # 'nt' es el identificador para Windows
    BASE_DIR = r"C:\Users\Victor\Documents\Tesis3D"
else: # 'posix' es para Linux, macOS, etc.
    BASE_DIR = r"/home/vit/Documentos/Tesis3D"

Nombre = "Sanchez"
# 2. Construir las rutas completas usando os.path.join
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "Videos", "Experimento_1", Nombre, f"{Nombre}_intento_8.mp4")

# --- ¡NUEVOS PARÁMETROS DE DEBUG! ---
INITIAL_CONFIDENCE_X100 = 50 # (Representa 0.50)
# ⚠️ ¡NUEVO! Umbral para guardar frames "difíciles" (Baja Confianza)
INITIAL_SAVE_THRESHOLD_X100 = 75 # (Representa 0.75) 
# ------------------------------------

# --- FUNCIONES DE UTILIDAD (Sin cambios) ---

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

def optimize_contours_by_angle(contours):
    # ... (Sin cambios) ...
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

# --- FUNCIÓN PRINCIPAL DE DEBUG (Modificada para Guardar Frames) ---
def debug_yolo_roi_processing(video_path, model_path):
    
    # --- ¡NUEVO! Definir y crear carpeta de guardado ---
    DEBUG_SAVE_DIR = os.path.join(BASE_DIR, "frames_para_retrain", Nombre)
    os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
    print(f"✅ Guardando frames para re-entrenar en: {DEBUG_SAVE_DIR}")
    # --- --- ---

    # --- 1. Cargar Modelo y Video ---
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error cargando el modelo YOLO: {e}"); return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video: {video_path}"); return
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30
    frame_delay = int(1000 / fps)

    print(f"Debug Pipeline YOLO: {video_path}")
    print("Ajusta los sliders. 'Confianza' (filtro) y 'Guardar < Conf' (umbral de guardado).")
    print("Press 'q' to quit, 'space' to pause.")

    window_name = "Debug | YOLO ROI (Azul) -> Fit (Amarillo)"
    cv2.namedWindow(window_name)
    
    # --- Sliders ---
    cv2.createTrackbar("Confianza (x100)", window_name, INITIAL_CONFIDENCE_X100, 100, on_trackbar)
    # --- ¡NUEVO SLIDER DE GUARDADO! ---
    cv2.createTrackbar("Guardar < Conf (x100)", window_name, INITIAL_SAVE_THRESHOLD_X100, 100, on_trackbar)


    while True:
        start_time = time.time()
        # --- ¡NUEVO! Banderas de estado por frame ---
        should_save_frame = False 
        save_reason = ""          
        # --- --- ---
        
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        # --- ¡NUEVO! Obtener Nro de frame ---
        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # --- 2. Preprocessing ---
        frame_cropped = crop_to_aspect_ratio(frame)
        # --- ¡NUEVO! Guardar una copia limpia para el dataset ---
        frame_to_save = frame_cropped.copy()
        # --- --- ---
        
        h_frame, w_frame = frame_cropped.shape[:2]
        
        # Crear paneles de visualización
        debug_panel_width = 320
        debug_panel_height = h_frame # 480
        debug_panel = np.zeros((debug_panel_height, debug_panel_width, 3), dtype=np.uint8)
        roi_zoom_h = debug_panel_height // 2 # 240
        mask_zoom_h = debug_panel_height // 2 # 240

        # --- 3. Obtener Confianza y Ejecutar YOLO ---
        current_conf_x100 = cv2.getTrackbarPos("Confianza (x100)", window_name)
        current_conf = current_conf_x100 / 100.0
        
        # --- ¡NUEVO! Obtener umbral de guardado ---
        current_save_conf_x100 = cv2.getTrackbarPos("Guardar < Conf (x100)", window_name)
        current_save_conf = current_save_conf_x100 / 100.0
        
        results = model.track(frame_cropped, persist=True, verbose=False, conf=current_conf)

        final_ellipse = None
        best_box = None
        max_conf_found = 0.0 

        # --- 4. Encontrar la mejor caja (ROI) ---
        if results[0].boxes:
            for box in results[0].boxes:
                if box.conf[0] > max_conf_found:
                    max_conf_found = box.conf[0]
                    best_box = box
        
        # --- 5. Si se encuentra un ROI, procesarlo ---
        if best_box is not None:
            # 5a. Extraer y dibujar el ROI
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)
            
            cv2.rectangle(frame_cropped, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            if x1 < x2 and y1 < y2:
                # 5b. Procesamiento dentro del ROI
                roi = frame_cropped[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours_roi, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if contours_roi:
                    best_pupil_contour_roi = max(contours_roi, key=cv2.contourArea)
                    best_pupil_contour_abs = best_pupil_contour_roi + (x1, y1)
                    optimized_contour = optimize_contours_by_angle([best_pupil_contour_abs])
                    
                    try:
                        if len(optimized_contour) >= 5:
                            final_ellipse = cv2.fitEllipse(optimized_contour)
                        elif len(best_pupil_contour_abs) >= 5:
                            final_ellipse = cv2.fitEllipse(best_pupil_contour_abs)
                    except cv2.error:
                        final_ellipse = None
                        
                    # 5e. Preparar visualización del panel de debug
                    roi_zoom = cv2.resize(roi, (debug_panel_width, roi_zoom_h))
                    debug_panel[0:roi_zoom_h, :] = roi_zoom
                    binary_roi_bgr = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(binary_roi_bgr, [best_pupil_contour_roi], -1, (0, 255, 0), 1)
                    mask_zoom = cv2.resize(binary_roi_bgr, (debug_panel_width, mask_zoom_h))
                    debug_panel[roi_zoom_h:debug_panel_height, :] = mask_zoom
        
        # --- 6. ¡NUEVO! Lógica de Guardado de Frames ---
        if final_ellipse is None:
            should_save_frame = True
            # Distinguir por qué falló
            if best_box is None:
                save_reason = "NO_ROI" # YOLO no encontró nada
            else:
                save_reason = "NO_ELIPSE_FIT" # YOLO encontró ROI, pero OpenCV falló
        
        # Se encontró elipse, pero la confianza del ROI de YOLO fue baja
        elif max_conf_found < current_save_conf: 
            should_save_frame = True
            save_reason = f"BAJA_CONF_{max_conf_found:.2f}"
        # --- --- ---

        # --- 7. Visualización Final ---
        
        # Dibujar Elipse final (AMARILLO) en el frame principal
        if final_ellipse is not None:
            cv2.ellipse(frame_cropped, final_ellipse, (0, 255, 255), 2) # Yellow
            
        # Combinar frame principal y panel de debug
        combined_view = np.hstack((frame_cropped, debug_panel))
        
        # --- Textos de información (ACTUALIZADO) ---
        conf_text = f"Conf. Slider: {current_conf:.2f}"
        save_conf_text = f"Guardar < {current_save_conf:.2f}" # ¡NUEVO!
        cv2.putText(combined_view, conf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_view, save_conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2) # Naranja

        if best_box is not None:
             found_text = f"Conf. Detectada: {max_conf_found:.2f}"
             cv2.putText(combined_view, found_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Textos de ayuda
        cv2.putText(combined_view, "Original + YOLO ROI (Azul) + Fit (Am)", (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(combined_view, "Zoom ROI (Arriba)", (w_frame + 10, roi_zoom_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(combined_view, "Zoom Mascara + Contorno (Verde)", (w_frame + 10, debug_panel_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # --- ¡NUEVO! Acción de guardado y texto en pantalla ---
        if should_save_frame:
            # 1. Mostrar texto en pantalla
            save_text = f"GUARDANDO ({save_reason})"
            # Centrar texto
            text_size, _ = cv2.getTextSize(save_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = (w_frame - text_size[0]) // 2
            text_y = (h_frame + text_size[1]) // 2
            cv2.putText(combined_view, save_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 2. Guardar el frame (la copia limpia 'frame_to_save')
            frame_filename = f"frame_{current_frame_num:06d}_{save_reason}.jpg"
            save_path = os.path.join(DEBUG_SAVE_DIR, frame_filename)
            
            # Guardamos la copia que no tiene dibujos
            cv2.imwrite(save_path, frame_to_save) 
        # --- --- ---

        cv2.imshow(window_name, combined_view)

        # --- 8. Control de Teclado ---
        processing_time = time.time() - start_time
        wait_time = max(1, int(frame_delay - (processing_time * 1000)))
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'): print("Processing stopped by user."); break
        elif key == ord(' '): cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# --- ENTRY POINT ---
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH): 
        print(f"Error: Video file not found at: {VIDEO_PATH}")
    elif not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model file not found at: {YOLO_MODEL_PATH}")
    else:
        debug_yolo_roi_processing(VIDEO_PATH, YOLO_MODEL_PATH)