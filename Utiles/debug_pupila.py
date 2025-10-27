import cv2
import numpy as np
import os
import time
import math # Añadido de nuevo para la optimización de ángulos

# --- PARÁMETROS DE PREPROCESAMIENTO ---
GAUSSIAN_KERNEL_SIZE = (7, 7)
CLAHE_CLIP_LIMIT = 1.0 # Fijo
INITIAL_THRESHOLD_VALUE = 85 # Value from your code
INITIAL_KERNEL_N = 2 # N=2 -> (2*2)+1 = 5. Usaremos un kernel de 5x5

# --- PARÁMETROS DE FILTRADO (AHORA VALORES INICIALES PARA TRACKBARS) ---
INITIAL_MIN_PUPIL_AREA = 1500 # Value from your code (changed from previous)
INITIAL_MAX_PUPIL_AREA = 15000  # Value from your code (changed from previous)
# Define ranges for trackbars
MAX_SLIDER_MIN_AREA = 5000
MAX_SLIDER_MAX_AREA = 30000

# --- NUEVO PARÁMETRO DE ROBUSTEZ ---
MIN_ELLIPTICAL_FIT_RATIO = 0.8 # Value from your code
MAX_ELLIPTICAL_FIT_RATIO = 1.10 # <<<--- NUEVO LÍMITE SUPERIOR CON TOLERANCIA
# ------------------------------------
VIDEO_PATH = r"/home/vit/Documentos/Tesis3D/Videos/Experimento_1/Chirinos/ROI_videos_640x480/Chirinos_intento_1_ROI_640x480.mp4" # <-- CHANGE THIS PATH
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

def find_darkest_3x3_center(image):
    min_sum = 255 * 9 + 1; min_loc = (1, 1)
    H, W = image.shape
    if H < 3 or W < 3: return (W // 2, H // 2) if H > 0 and W > 0 else (0, 0)
    for y in range(H - 2):
        for x in range(W - 2):
            s = int(image[y, x])   + int(image[y, x+1])   + int(image[y, x+2]) + \
                int(image[y+1, x]) + int(image[y+1, x+1]) + int(image[y+1, x+2]) + \
                int(image[y+2, x]) + int(image[y+2, x+1]) + int(image[y+2, x+2])
            if s < min_sum: min_sum = s; min_loc = (x + 1, y + 1)
    return min_loc

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

def on_trackbar(val): pass

# --- FUNCIÓN PRINCIPAL DE DEBUG (Modificada) ---
def debug_full_frame_processing(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error opening video: {video_path}"); return
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps > 0 else 30
    frame_delay = int(1000 / fps)

    print(f"Processing video (Full Frame): {video_path}")
    print("Adjust Threshold, Kernel Size, Min Area, and Max Area sliders.")
    print("Press 'q' to quit, 'space' to pause.")

    window_name = "Debug Full Frame | Area & Elliptical Fit Filter"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Threshold", window_name, INITIAL_THRESHOLD_VALUE, 255, on_trackbar)
    cv2.createTrackbar("Kernel Size (N)", window_name, INITIAL_KERNEL_N, 5, on_trackbar)
    cv2.createTrackbar("Min Area", window_name, INITIAL_MIN_PUPIL_AREA, MAX_SLIDER_MIN_AREA, on_trackbar)
    cv2.createTrackbar("Max Area", window_name, INITIAL_MAX_PUPIL_AREA, MAX_SLIDER_MAX_AREA, on_trackbar)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        # --- 1. Preprocessing ---
        frame_cropped = crop_to_aspect_ratio(frame)
        frame_blurred = cv2.GaussianBlur(frame_cropped, GAUSSIAN_KERNEL_SIZE, 0)
        gray_frame_original = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)

        # --- 2. CLAHE ---
        gray_frame_clahe = clahe.apply(gray_frame_original)

        # --- 3. Darkest Point (for visualization only) ---
        dark_point = find_darkest_3x3_center(gray_frame_clahe)

        # --- 4. Binarization & Morphology ---
        current_threshold = cv2.getTrackbarPos("Threshold", window_name)
        current_kernel_n = max(1, cv2.getTrackbarPos("Kernel Size (N)", window_name))
        kernel_size = (current_kernel_n * 2) + 1
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        thresholded_image_raw = apply_fixed_binary_threshold(gray_frame_clahe, current_threshold)
        thresholded_image_closed = cv2.morphologyEx(thresholded_image_raw, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        thresholded_image_final = cv2.morphologyEx(thresholded_image_closed, cv2.MORPH_OPEN, morph_kernel, iterations=1)

        # --- Read current Area trackbar values ---
        current_min_area = cv2.getTrackbarPos("Min Area", window_name)
        current_max_area = cv2.getTrackbarPos("Max Area", window_name)
        if current_min_area > current_max_area: current_min_area = current_max_area

        # --- 5. Contour Filtering ---
        contours, _ = cv2.findContours(thresholded_image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Pre-filter by Area ---
        contours_in_area_range = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if current_min_area <= contour_area <= current_max_area:
                contours_in_area_range.append(contour)

        best_pupil_contour = None
        best_fit_ratio = 0.0
        best_contour_area = 0.0

        # --- Main Loop: Iterate ONLY over valid area contours ---
        for contour in contours_in_area_range:
            # --- Elliptical Shape Filter ---
            if len(contour) < 5: continue
            try:
                fitted_ellipse = cv2.fitEllipse(contour)
                (width, height) = fitted_ellipse[1]
                if width <= 0 or height <= 0: continue
                ellipse_area = (np.pi / 4.0) * width * height
                if ellipse_area <= 0: continue
                contour_area = cv2.contourArea(contour)
                fit_ratio = contour_area / ellipse_area

                # --- CONDITION MODIFIED ---
                # Check if ratio is valid (MIN < ratio <= MAX) AND better than current best
                if MIN_ELLIPTICAL_FIT_RATIO < fit_ratio <= MAX_ELLIPTICAL_FIT_RATIO and fit_ratio > best_fit_ratio:
                    best_fit_ratio = fit_ratio
                    best_pupil_contour = contour
                    best_contour_area = contour_area
            except cv2.error: continue
        # --- END OF LOOP ---

        # Prepare mask visualization
        final_bgr = cv2.cvtColor(thresholded_image_final, cv2.COLOR_GRAY2BGR)

        # --- 6. Optimization & Drawing ---
        cv2.drawContours(final_bgr, contours_in_area_range, -1, (255, 0, 0), 1) # Blue
        if best_pupil_contour is not None:
            cv2.drawContours(final_bgr, [best_pupil_contour], -1, (0, 255, 0), 1) # Green
            optimized_contour = optimize_contours_by_angle([best_pupil_contour])
            try:
                final_ellipse = None
                if len(optimized_contour) >= 5:
                    final_ellipse = cv2.fitEllipse(optimized_contour)
                else:
                    final_ellipse = cv2.fitEllipse(best_pupil_contour)
                if final_ellipse:
                    cv2.ellipse(frame_cropped, final_ellipse, (0, 255, 255), 2) # Yellow
            except cv2.error: pass

        # Draw darkest point
        cv2.circle(frame_cropped, dark_point, 5, (0, 0, 255), -1) # Red

        # --- 7. Final Visualization ---
        combined_view = np.hstack((frame_cropped, final_bgr))
        cv2.putText(combined_view, f"Threshold: {current_threshold}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_view, f"Kernel Size: {kernel_size}x{kernel_size}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_view, f"Area Range: {current_min_area}-{current_max_area}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_view, f"Best Valid Fit: {best_fit_ratio:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_view, f"Best Contour Area: {best_contour_area:.0f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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