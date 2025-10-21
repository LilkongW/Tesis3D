import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pyautogui
import math
import time
from PuntoRojo import create_moving_point,move_point_to,destroy_moving_point
from Overlay import start_center_calibration

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2


MODEL_PATH = r'C:\Users\Victor\Documents\Tesis2\Trayectoria\Model'

# Configuración
filter_length = 10
gaze_length = 350
dibujar_landmarks = True
# Buffers para suavizado
combined_gaze_directions = deque(maxlen=filter_length)

# Buffers to store recent ray data
ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)
calibration_offset_yaw = 0
calibration_offset_pitch = 0


# Inicializar MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

Video_Path = r"C:\Users\Victor\Documents\Tesis2\Videos\Victor\Data_Victor_1.avi"

# Abrir webcam
cap = cv2.VideoCapture(Video_Path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Índices de landmarks de la nariz (para tracking estable)
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

LANDMARKS = {
    "left": 234,
    "right": 454,
    "top": 10,
    "bottom": 152,
    "front": 1,
}

FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# Variables de tracking de esferas oculares
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

R_ref_nose = [None]
base_radius = 20

def landmark_to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

def compute_scale(points_3d):
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0

def compute_coordinate_box(face_landmarks, indices, ref_matrix_container):
    from scipy.spatial.transform import Rotation as Rscipy
    
    # Extraer posiciones 3D
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])
    
    center = np.mean(points_3d, axis=0)
    
    # PCA para orientación
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1
    
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()
    
    # Estabilizar rotación
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1
    
    return center, R_final, points_3d

def draw_gaze_ray(frame, eye_center, iris_center, eye_radius, color, length):
    # Vector de mirada
    gaze_direction = iris_center - eye_center
    gaze_direction /= np.linalg.norm(gaze_direction)
    gaze_endpoint = eye_center + gaze_direction * length
    
    # Dibujar línea de mirada
    cv2.line(frame, 
             tuple(int(v) for v in eye_center[:2]), 
             tuple(int(v) for v in gaze_endpoint[:2]), 
             color, 2)
    
    # Offset del iris
    iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)
    
    # Segmento trasero (detrás del iris)
    cv2.line(frame,
             (int(eye_center[0]), int(eye_center[1])),
             (int(iris_offset[0]), int(iris_offset[1])),
             color, 1)
    
    # Segmento frontal (delante del iris)
    cv2.line(frame,
             (int(iris_offset[0]), int(iris_offset[1])),
             (int(gaze_endpoint[0]), int(gaze_endpoint[1])),
             color, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        landmarks_frame = np.zeros_like(frame)  # Blank black frame

        outline_pts = []

        #Aqui dibujamos la caja de la cara y los puntos de referencia

        # Draw all landmarks as single white pixels
        for i, landmark in enumerate(face_landmarks):
            pt = landmark_to_np(landmark, w, h)
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                color = (155, 155, 155) if i in FACE_OUTLINE_INDICES else (255, 25, 10)

        # Highlight bounding landmarks in pink
        key_points = {}
        for name, idx in LANDMARKS.items():
            pt = landmark_to_np(face_landmarks[idx], w, h)
            key_points[name] = pt
            x, y = int(pt[0]), int(pt[1])


        # Extract points
        left = key_points["left"]
        right = key_points["right"]
        top = key_points["top"]
        bottom = key_points["bottom"]
        front = key_points["front"]


        # Oriented axes based on head geometry
        right_axis = (right - left)
        right_axis /= np.linalg.norm(right_axis)

        up_axis = (top - bottom)
        up_axis /= np.linalg.norm(up_axis)

        forward_axis = np.cross(right_axis, up_axis)
        forward_axis /= np.linalg.norm(forward_axis)

        # Flip to ensure forward vector comes out of the face
        forward_axis = -forward_axis

        # Compute center of the head
        center = (left + right + top + bottom + front) / 5

        # Half-sizes (width, height, depth)
        half_width = np.linalg.norm(right - left) / 2
        half_height = np.linalg.norm(top - bottom) / 2
        half_depth = 80  # can be tuned or calculated if you have a back landmark

        # Generate cube corners in face-aligned space
        def corner(x_sign, y_sign, z_sign):
            return (center
                    + x_sign * half_width * right_axis
                    + y_sign * half_height * up_axis
                    + z_sign * half_depth * forward_axis)

        cube_corners = [
            corner(-1, 1, -1),   # top-left-front
            corner(1, 1, -1),    # top-right-front
            corner(1, -1, -1),   # bottom-right-front
            corner(-1, -1, -1),  # bottom-left-front
            corner(-1, 1, 1),    # top-left-back
            corner(1, 1, 1),     # top-right-back
            corner(1, -1, 1),    # bottom-right-back
            corner(-1, -1, 1)    # bottom-left-back
        ]

        # Projection function
        def project(pt3d):
            return int(pt3d[0]), int(pt3d[1])

        # Draw wireframe cube
        cube_corners_2d = [project(pt) for pt in cube_corners]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # back face
            (0, 4), (1, 5), (2, 6), (3, 7)   # sides
        ]

        # Update smoothing buffers
        ray_origins.append(center)
        ray_directions.append(forward_axis)

        # Compute averaged ray origin and direction
        avg_origin = np.mean(ray_origins, axis=0)
        avg_direction = np.mean(ray_directions, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)  # normalize

        # Reference forward direction (camera looking straight ahead)
        reference_forward = np.array([0, 0, -1])  # Z-axis into the screen

        # Horizontal (yaw) angle from reference (project onto XZ plane)
        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        xz_proj /= np.linalg.norm(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad  # left is negative

        # Vertical (pitch) angle from reference (project onto YZ plane)
        yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
        yz_proj /= np.linalg.norm(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad  # up is positive

        #Specify 
        # Convert to degrees and re-center around 0
        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        #this results in the center being 180, +10 left = -170, +10 right = +170

        #convert left rotations to 0-180
        if yaw_deg < 0:
            yaw_deg = abs(yaw_deg)
        elif yaw_deg < 180:
            yaw_deg = 360 - yaw_deg

        if pitch_deg < 0:
            pitch_deg = 360 + pitch_deg

        raw_yaw_deg = yaw_deg
        raw_pitch_deg = pitch_deg
        
        #yaw is now converted to 90 (looking directly left) to 270 (looking directly right), wrt camera
        #pitch is now converted to 90 (looking straight down) and 270 (looking straight up), wrt camera
        #print(f"Angles: yaw={yaw_deg}, pitch={pitch_deg}")

        #specify degrees at which screen border will be reached
        yawDegrees = 20 # x degrees left or right
        pitchDegrees = 10 # x degrees up or down
        
        # leftmost pixel position must correspond to 180 - yaw degrees
        # rightmost pixel position must correspond to 180 + yaw degrees
        # topmost pixel position must correspond to 180 + pitch degrees
        # bottommost pixel position must correspond to 180 - pitch degrees

        # Apply calibration offsets
        yaw_deg += calibration_offset_yaw
        pitch_deg += calibration_offset_pitch

        # Map to full screen resolution
        screen_x = int(((yaw_deg - (180 - yawDegrees)) / (2 * yawDegrees)) * MONITOR_WIDTH)
        screen_y = int(((180 + pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)

        # Clamp screen position to monitor bounds
        if(screen_x < 10):
            screen_x = 10
        if(screen_y < 10):
            screen_y = 10
        if(screen_x > MONITOR_WIDTH - 10):
            screen_x = MONITOR_WIDTH - 10
        if(screen_y > MONITOR_HEIGHT - 10):
            screen_y = MONITOR_HEIGHT - 10

        #print(f"Screen position: x={screen_x}, y={screen_y}")

        # Draw smoothed ray
        ray_length = 2.5 * half_depth
        ray_end = avg_origin - avg_direction * ray_length
        
        # Dibujamos el rayo frontal 
        cv2.line(frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)
        cv2.line(landmarks_frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)

        #===========================================================================
        # Índices de iris
        left_iris_idx = 468
        right_iris_idx = 473
        left_iris = face_landmarks[left_iris_idx]
        right_iris = face_landmarks[right_iris_idx]
        
        # Calcular centro de cabeza y orientación
        head_center, R_final, nose_points_3d = compute_coordinate_box(
            face_landmarks, nose_indices, R_ref_nose
        )
        
        # Posiciones 3D de iris
        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])
        
        # === CALIBRACIÓN AUTOMÁTICA EN EL PRIMER FRAME ===
        if not (left_sphere_locked and right_sphere_locked):

            start_center_calibration ()

            time.sleep(1)

            current_nose_scale = compute_scale(nose_points_3d)
            
            # Lock ojo izquierdo
            left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
            camera_dir_world = np.array([0, 0, 1])
            camera_dir_local = R_final.T @ camera_dir_world
            left_sphere_local_offset += base_radius * camera_dir_local
            left_calibration_nose_scale = current_nose_scale
            left_sphere_locked = True
            
            # Lock ojo derecho
            right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
            right_sphere_local_offset += base_radius * camera_dir_local
            right_calibration_nose_scale = current_nose_scale
            right_sphere_locked = True
            
            print("[Calibración Automática] Esferas oculares bloqueadas.")

            calibration_offset_yaw = 180 - raw_yaw_deg
            calibration_offset_pitch = 180 - raw_pitch_deg

            print(f"[Calibración Automática] Offset Yaw: {calibration_offset_yaw}, Offset Pitch: {calibration_offset_pitch} Frente Calibrado")

        
        # === CALCULAR POSICIONES DE ESFERAS CON ESCALA ===
        current_nose_scale = compute_scale(nose_points_3d)
        
        scale_ratio_l = current_nose_scale / left_calibration_nose_scale
        scaled_offset_l = left_sphere_local_offset * scale_ratio_l
        sphere_world_l = head_center + R_final @ scaled_offset_l
        scaled_radius_l = int(base_radius * scale_ratio_l)
        
        scale_ratio_r = current_nose_scale / right_calibration_nose_scale
        scaled_offset_r = right_sphere_local_offset * scale_ratio_r
        sphere_world_r = head_center + R_final @ scaled_offset_r
        scaled_radius_r = int(base_radius * scale_ratio_r)
        
        # Dibujar esferas
        cv2.circle(frame, (int(sphere_world_l[0]), int(sphere_world_l[1])), 
                   scaled_radius_l, (255, 255, 25), 2)
        cv2.circle(frame, (int(sphere_world_r[0]), int(sphere_world_r[1])), 
                   scaled_radius_r, (25, 255, 255), 2)
        
        # === DIBUJAR RAYOS DE MIRADA INDIVIDUALES ===
        draw_gaze_ray(frame, sphere_world_l, iris_3d_left, scaled_radius_l, (155, 155, 25), 130)
        draw_gaze_ray(frame, sphere_world_r, iris_3d_right, scaled_radius_r, (25, 155, 155), 130)
        
        # === CALCULAR Y DIBUJAR RAYO PROMEDIO ===
        left_gaze_dir = iris_3d_left - sphere_world_l
        left_gaze_dir /= np.linalg.norm(left_gaze_dir)
        
        right_gaze_dir = iris_3d_right - sphere_world_r
        right_gaze_dir /= np.linalg.norm(right_gaze_dir)
        
        # Dirección combinada
        raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        raw_combined_direction /= np.linalg.norm(raw_combined_direction)
        
        # Buffer de suavizado
        combined_gaze_directions.append(raw_combined_direction)
        avg_combined_direction = np.mean(combined_gaze_directions, axis=0)
        avg_combined_direction /= np.linalg.norm(avg_combined_direction)
        
        # Dibujar rayo promedio (más grueso y de color distinto)
        combined_origin = (sphere_world_l + sphere_world_r) / 2
        combined_target = combined_origin + avg_combined_direction * gaze_length
        cv2.line(frame,
                (int(combined_origin[0]), int(combined_origin[1])),
                (int(combined_target[0]), int(combined_target[1])),
                (255, 255, 10), 3)
    
    cv2.imshow("Eye Tracking - Gaze Rays", frame)
    

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        dibujar_landmarks = False

    elif key == ord('s'):
        dibujar_landmarks = True

    elif key == ord('t'):
        #Aqui mapeareamos las posiciones predichas en la pantalla y moveriamos un punto rojo

        #Cargamos el modelo

        #Hacemos predicciones
        screen_x = None

        screen_y = None

        print (f"posicion x:{screen_x} posicion y:{screen_y}")

        #Primero creamos el punto
        create_moving_point ()

        #Movemos el punto con las predicciones 

        move_point_to(screen_x, screen_y)



destroy_moving_point()
cap.release()
cv2.destroyAllWindows()