# main_tracker_integrated.py

import cv2
import numpy as np
import os
import sys
import threading
import time
import math
import random

from gl_sphere import (
    start_gl_window, 
    update_sphere_rotation_signal,
    get_latest_rendered_image,
    app, 
    sphere_widget,
    gl_image_buffer
)

# =================================================================
# VARIABLES GLOBALES Y PARÁMETROS DE DETECCIÓN (ACTUALIZADOS)
# =================================================================

# --- PARÁMETROS DE FILTRADO DE CONTORNOS (Basados en Debug_Step_5) ---
GAUSSIAN_KERNEL_SIZE = (7, 7) 
CLAHE_CLIP_LIMIT = 1.0       # Fijo, de debug
FIXED_THRESHOLD_VALUE = 80   # Fijo, de debug
MORPH_KERNEL_SIZE = 5        # (N=2 -> 5x5), de debug
MIN_PUPIL_AREA = 1200 
MAX_PUPIL_AREA = 10000 
# MAX_ELLIPSE_RATIO ya no se usa

# --- PARÁMETRO DE ESTABILIDAD DEL MODELO ---
MAX_INTERSECTION_DISTANCE = 60 # Distancia máxima en píxeles que una intersección puede estar del centro del modelo previo.

# --- VARIABLES DE ESTADO ---
ray_lines = [] 
model_centers = []
stable_pupil_centers = [] # Para suavizar la posición de la pupila
max_rays = 100
prev_model_center_avg = (320, 240)
max_observed_distance = 230 # Radio fijo del círculo cian

# Variables para la comunicación entre hilos
gl_ready = threading.Event()

# --- OBJETOS GLOBALES DE CV (NUEVO) ---
# Crear una sola vez para eficiencia
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))


# =================================================================
# --- FUNCIONES DE LÓGICA DE DETECCIÓN (INTEGRADAS) ---
# =================================================================

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

# --- 'detect_and_select_pupil' FUE ELIMINADA, LA NUEVA LÓGICA ESTÁ EN 'process_frame_for_comparison' ---

def update_and_average_point(point_list, new_point, N):
    point_list.append(new_point)
    if len(point_list) > N:
        point_list.pop(0)
    if not point_list:
        return None
    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))
    return (avg_x, avg_y)

# --- 'find_line_intersection' REEMPLAZADA POR VERSIÓN MÁS ROBUSTA ---
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

def compute_average_intersection(frame, ray_lines, N, M, spacing):
    global stored_intersections, prev_model_center_avg
    stored_intersections = getattr(compute_average_intersection, 'stored_intersections', [])
    if len(ray_lines) < 2 or N < 2:
        return None
    height, width = frame.shape[:2]
    selected_lines = random.sample(ray_lines, min(N, len(ray_lines)))
    intersections = []
    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]
        if abs(line1[2] - line2[2]) >= 2:
            intersection = find_line_intersection(line1, line2)
            if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                dist = math.hypot(intersection[0] - prev_model_center_avg[0], intersection[1] - prev_model_center_avg[1])
                if dist < MAX_INTERSECTION_DISTANCE:
                    intersections.append(intersection)
                    stored_intersections.append(intersection)
    if len(stored_intersections) > M:
        stored_intersections = stored_intersections[-M:]
    compute_average_intersection.stored_intersections = stored_intersections
    if not stored_intersections:
        return None
    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])
    return (int(avg_x), int(avg_y))

# =================================================================
# THREAD DE VISUALIZACIÓN 3D
# =================================================================
def run_gl_thread():
    """Inicia la aplicación PyQt/OpenGL en un hilo separado."""
    global app
    app = start_gl_window()
    gl_ready.set()
    sys.exit(app.exec_())

# =================================================================
# CÓDIGO DE PROCESAMIENTO (AHORA USA LA LÓGICA INTEGRADA)
# =================================================================
def process_frame_for_comparison(frame):
    """Procesa un frame usando la lógica robusta y devuelve datos para visualización."""
    global ray_lines, model_centers, max_rays, prev_model_center_avg, max_observed_distance, stable_pupil_centers
    global clahe, morph_kernel # Usar los objetos globales
    
    # 1. Pre-procesamiento
    frame = crop_to_aspect_ratio(frame)
    frame_orig_shape = frame.shape
    frame_blurred = cv2.GaussianBlur(frame, GAUSSIAN_KERNEL_SIZE, 0)
    gray_frame_original = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    
    # --- INICIO DE LA NUEVA LÓGICA DE DETECCIÓN (DE DEBUG_STEP_5) ---
    
    # 2. Aplicar CLAHE
    gray_frame_clahe = clahe.apply(gray_frame_original)
    
    # 3. Encontrar punto ancla (el más oscuro)
    dark_point = find_darkest_2x2(gray_frame_clahe) # (x, y)
    
    # 4. Binarización y Morfología
    thresholded_image_raw = apply_fixed_binary_threshold(gray_frame_clahe, FIXED_THRESHOLD_VALUE)
    thresholded_image_closed = cv2.morphologyEx(thresholded_image_raw, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    thresholded_image_final = cv2.morphologyEx(thresholded_image_closed, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # 5. Filtrado de Contorno por Punto Ancla
    contours, _ = cv2.findContours(thresholded_image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pupil_contour = None
    for contour in contours:
        
        # Filtro 1: Área
        contour_area = cv2.contourArea(contour)
        if not (MIN_PUPIL_AREA <= contour_area <= MAX_PUPIL_AREA):
            continue 

        # Filtro 2: Punto Ancla
        if cv2.pointPolygonTest(contour, dark_point, False) >= 0:
            pupil_contour = contour
            break # Encontramos el que queríamos

    # --- FIN DE LA NUEVA LÓGICA DE DETECCIÓN ---
    
    final_rotated_rect = None
    center_x, center_y = None, None
    pupil_center_smoothed = None
    dark_point_to_draw = dark_point # Guardar el ancla para dibujarla
    
    if pupil_contour is not None and len(pupil_contour) >= 5:
        # Optimizar contorno y ajustar elipse
        optimized_contour = optimize_contours_by_angle([pupil_contour])
        
        try:
            if len(optimized_contour) >= 5:
                ellipse = cv2.fitEllipse(optimized_contour)
            else:
                # Fallback si la optimización falló
                ellipse = cv2.fitEllipse(pupil_contour)
            
            final_rotated_rect = ellipse
            center_x, center_y = map(int, final_rotated_rect[0])
            
            # Suavizar la posición de la pupila
            pupil_center_smoothed = update_and_average_point(stable_pupil_centers, (center_x, center_y), N=3)
            
            # Almacenar el rayo (elipse) para el cálculo del centro del modelo
            ray_lines.append(final_rotated_rect)
            if len(ray_lines) > max_rays:
                ray_lines = ray_lines[-max_rays:]
        
        except cv2.error:
            # fitEllipse puede fallar
            final_rotated_rect = None
            pupil_center_smoothed = None
    
    # 6. Calcular centro del modelo (esfera cian)
    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5)
    
    # Suavizar el centro del modelo
    if model_center is not None:
        model_center_average = update_and_average_point(model_centers, model_center, 800)
        prev_model_center_avg = model_center_average
    else:
        model_center_average = prev_model_center_avg
    
    # 7. Dibujar en el frame original
    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)
    
    # Dibujar el ancla 2x2 (punto rojo)
    #cv2.circle(frame, dark_point_to_draw, 5, (0, 0, 255), -1)

    if final_rotated_rect is not None and pupil_center_smoothed is not None:
        px, py = pupil_center_smoothed
        cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2) # Elipse amarilla
        cv2.line(frame, model_center_average, (px, py), (255, 150, 50), 2)
        
        # Línea extendida
        dx = px - model_center_average[0]
        dy = py - model_center_average[1]
        extended_x = int(model_center_average[0] + 2 * dx)
        extended_y = int(model_center_average[1] + 2 * dy)
        cv2.line(frame, (px, py), (extended_x, extended_y), (200, 255, 0), 3)

    return frame, model_center_average, pupil_center_smoothed, frame_orig_shape

# =================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO DE VIDEO
# =================================================================
def process_video_comparison(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video '{video_path}' no existe.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video '{video_path}'.")
        return
    
    gl_thread = threading.Thread(target=run_gl_thread)
    gl_thread.start()
    
    print("⏳ Esperando que la ventana 3D de OpenGL se inicie...")
    gl_ready.wait()
    print("✅ Ventana 3D lista.")

    print(f"Procesando: {video_path}")
    print("\nControles:\n   'q' - Salir\n   'ESPACIO' - Pausar/Reanudar\n" + "-"*60)
    
    frame_number = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n⏹️   Fin del video alcanzado")
                    break
                
                try:
                    # 1. Procesar frame y obtener los datos
                    processed_frame, model_center, pupil_center, frame_shape = \
                        process_frame_for_comparison(frame)
                    
                    # 2. ENVIAR los datos al hilo de OpenGL
                    if pupil_center and model_center:
                        update_sphere_rotation_signal(
                            pupil_center[0], pupil_center[1], 
                            model_center[0], model_center[1], 
                            frame_shape[1], frame_shape[0]
                        )
                    
                    # 3. RECUPERAR la última imagen renderizada de OpenGL
                    gl_image = get_latest_rendered_image()
                        
                    # 4. COMBINAR AMBAS IMÁGENES
                    if gl_image is not None:
                        clean_view = cv2.cvtColor(gl_image, cv2.COLOR_RGB2BGR)
                        h_cv, w_cv, _ = processed_frame.shape
                        h_gl, w_gl, _ = clean_view.shape

                        if h_cv != h_gl:
                            new_w_gl = int(w_gl * h_cv / h_gl)
                            clean_view = cv2.resize(clean_view, (new_w_gl, h_cv), interpolation=cv2.INTER_LINEAR)
                            
                        combined_frame = np.hstack((processed_frame, clean_view))
                        cv2.imshow("Eye Tracking: Real vs. 3D Model", combined_frame)
                    else:
                        cv2.imshow("Eye Tracking: Real vs. 3D Model", processed_frame)
                    
                    frame_number += 1
                    if frame_number % 30 == 0:
                        print(f"📊 Frame {frame_number} procesado", end='\r')
                            
                except Exception as e:
                    print(f"\n⚠️   Error procesando frame {frame_number}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                print("\n🛑 Detenido por usuario")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"\n{'⏸️   PAUSADO' if paused else '▶️   REPRODUCIENDO'}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if app is not None:
            app.quit()
        if gl_thread.is_alive():
             gl_thread.join(timeout=1)
        print("\n✅ Procesamiento completado")
        print(f"📊 Total frames: {frame_number}")

def main():
    print("="*60 + "\nCOMPARADOR DE EYE TRACKING - OJO REAL VS VIRTUAL\n" + "="*60)
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = r"C:\Users\Victor\Documents\Tesis3D\Videos\Experimento_1\Victor\ROI_videos_640x480\grabacion_experimento_ESP32CAM_1_ROI_640x480.mp4"
    
    process_video_comparison(video_path)

if __name__ == "__main__":
    main()