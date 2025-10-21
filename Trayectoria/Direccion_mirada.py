import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pyautogui
import math
from Overlay import start_center_calibration,show_multipoint
import time

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2

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

Video_Path = r"c:\Users\Victor\Downloads\video_20251009_130204.mp4"

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
paused = False


def convert_gaze_to_facial_coordinates(gaze_direction, center, right_axis, up_axis, forward_axis):
    """
    Convierte un vector de dirección de mirada al sistema de coordenadas facial
    
    Args:
        gaze_direction: Vector de dirección de mirada en coordenadas mundiales
        center: Centro del sistema facial
        right_axis: Eje X (derecha)
        up_axis: Eje Y (arriba)  
        forward_axis: Eje Z (frente)
    
    Returns:
        Vector de mirada en coordenadas faciales [x, y, z]
    """
    # Crear matriz de rotación desde ejes faciales
    # Los ejes ya son ortonormales (right_axis × up_axis = forward_axis)
    R_facial = np.column_stack([right_axis, up_axis, forward_axis])
    
    # Convertir dirección de mirada al sistema facial
    gaze_facial = R_facial.T @ gaze_direction
    
    return gaze_facial

def gaze_to_angles_facial_coordinates(gaze_facial):
    """
    Convierte vector de mirada en coordenadas faciales a ángulos
    
    Returns:
        yaw_deg: Ángulo horizontal (- izquierda, + derecha)
        pitch_deg: Ángulo vertical (- abajo, + arriba)
    """
    # Normalizar el vector
    gaze_norm = gaze_facial / np.linalg.norm(gaze_facial)
    
    # Calcular ángulos
    # Yaw (horizontal): ángulo en plano XZ
    yaw_rad = math.atan2(gaze_norm[0], gaze_norm[2])  # atan2(X, Z)
    yaw_deg = np.degrees(yaw_rad)
    
    # Pitch (vertical): ángulo en plano YZ  
    pitch_rad = math.asin(gaze_norm[1])  # asin(Y)
    pitch_deg = np.degrees(pitch_rad)
    
    return yaw_deg, pitch_deg

# Función project global
def project(pt3d):
    return int(pt3d[0]), int(pt3d[1])

def convert_vectors_to_head_space(vectors_data, R_final, head_center):
    """
    Convierte todos los vectores al sistema de coordenadas de la cabeza
    
    Sistema de cabeza:
    - Origen: centro de la cabeza
    - X: derecha de la cabeza
    - Y: arriba de la cabeza  
    - Z: frente de la cabeza
    """
    head_space_vectors = {}
    
    # Convertir POSICIONES al sistema de cabeza
    head_space_vectors['sphere_world_l_head'] = R_final.T @ (vectors_data['sphere_world_l'] - head_center)
    head_space_vectors['sphere_world_r_head'] = R_final.T @ (vectors_data['sphere_world_r'] - head_center)
    
    # Convertir VECTORES DIRECCIÓN al sistema de cabeza
    head_space_vectors['left_gaze_head'] = R_final.T @ vectors_data['left_gaze_dir']
    head_space_vectors['right_gaze_head'] = R_final.T @ vectors_data['right_gaze_dir']
    head_space_vectors['avg_gaze_head'] = R_final.T @ vectors_data['avg_combined_direction']
    head_space_vectors['head_forward_head'] = R_final.T @ vectors_data['avg_direction']
    
    # El vector forward de referencia en sistema cabeza es [0, 0, 1]
    head_space_vectors['head_forward_ref'] = np.array([0, 0, 1])
    
    return head_space_vectors

def compute_relative_angles(head_space_vectors):
    """Calcula ángulos entre vectores de mirada y la dirección de la cabeza"""
    angles = {}
    
    # Vector forward de referencia
    head_forward = head_space_vectors['head_forward_ref']
    
    # Calcular ángulos con la dirección frontal de la cabeza
    vectors_to_check = {
        'left_gaze': head_space_vectors['left_gaze_head'],
        'right_gaze': head_space_vectors['right_gaze_head'],
        'avg_gaze': head_space_vectors['avg_gaze_head'],
        'head_actual': head_space_vectors['head_forward_head']
    }
    
    for name, vector in vectors_to_check.items():
        # Ángulo total
        dot_product = np.clip(np.dot(vector, head_forward), -1.0, 1.0)
        angles[f'{name}_angle_total'] = np.degrees(np.arccos(dot_product))
        
        # Componentes horizontales (proyección en plano XZ)
        horizontal_vec = np.array([vector[0], 0, vector[2]])
        if np.linalg.norm(horizontal_vec) > 0.001:
            horizontal_vec = horizontal_vec / np.linalg.norm(horizontal_vec)
            horizontal_ref = np.array([0, 0, 1])  # Z positivo es frente
            dot_horizontal = np.clip(np.dot(horizontal_vec, horizontal_ref), -1.0, 1.0)
            angle_horizontal = np.degrees(np.arccos(dot_horizontal))
            # Determinar izquierda/derecha
            if vector[0] < 0:  # Componente X negativa = izquierda
                angles[f'{name}_angle_horizontal'] = -angle_horizontal
            else:
                angles[f'{name}_angle_horizontal'] = angle_horizontal
        else:
            angles[f'{name}_angle_horizontal'] = 0.0
        
        # Componentes verticales (proyección en plano YZ)
        vertical_vec = np.array([0, vector[1], vector[2]])
        if np.linalg.norm(vertical_vec) > 0.001:
            vertical_vec = vertical_vec / np.linalg.norm(vertical_vec)
            vertical_ref = np.array([0, 0, 1])  # Z positivo es frente
            dot_vertical = np.clip(np.dot(vertical_vec, vertical_ref), -1.0, 1.0)
            angle_vertical = np.degrees(np.arccos(dot_vertical))
            # Determinar arriba/abajo
            if vector[1] < 0:  # Componente Y negativa = arriba
                angles[f'{name}_angle_vertical'] = angle_vertical
            else:
                angles[f'{name}_angle_vertical'] = -angle_vertical
        else:
            angles[f'{name}_angle_vertical'] = 0.0
    
    return angles

def draw_head_coordinate_system(frame, head_space_vectors, angles, position="top-right", size=80):
    """
    Dibuja sistema de coordenadas de cabeza con vectores convertidos
    """
    h, w = frame.shape[:2]
    
    # Posición en pantalla
    if position == "top-right":
        origin_x = w - size - 30
        origin_y = size + 60
    else:
        origin_x = size + 30
        origin_y = size + 60
    
    origin = np.array([origin_x, origin_y])
    vector_scale = size * 0.6
    
    # Fondo semitransparente
    overlay = frame.copy()
    bg_width = size + 80
    bg_height = size + 100
    cv2.rectangle(overlay, 
                 (origin_x - bg_width//2, origin_y - bg_height//2),
                 (origin_x + bg_width//2, origin_y + bg_height//2),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Ejes del sistema de cabeza
    # X: Rojo (derecha de la cabeza)
    x_end = origin + np.array([vector_scale, 0])
    cv2.arrowedLine(frame, tuple(origin.astype(int)), tuple(x_end.astype(int)), 
                   (0, 0, 255), 2, tipLength=0.2)
    cv2.putText(frame, "X", (x_end[0] + 5, x_end[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Y: Verde (arriba de la cabeza - negativo en coordenadas de imagen)
    y_end = origin + np.array([0, -vector_scale])
    cv2.arrowedLine(frame, tuple(origin.astype(int)), tuple(y_end.astype(int)), 
                   (0, 255, 0), 2, tipLength=0.2)
    cv2.putText(frame, "Y", (y_end[0], y_end[1] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Z: Azul (frente de la cabeza)
    z_end = origin + np.array([0, vector_scale])
    cv2.arrowedLine(frame, tuple(origin.astype(int)), tuple(z_end.astype(int)), 
                   (255, 0, 0), 2, tipLength=0.2)
    cv2.putText(frame, "Z", (z_end[0], z_end[1] + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Dibujar vectores EN SISTEMA DE CABEZA
    vectors_to_draw = [
        (head_space_vectors['left_gaze_head'], "L", (255, 200, 0)),
        (head_space_vectors['right_gaze_head'], "R", (255, 100, 0)),
        (head_space_vectors['avg_gaze_head'], "Avg", (255, 255, 0)),
        (head_space_vectors['head_forward_head'], "Head", (0, 255, 255))
    ]
    
    for vec, label, color in vectors_to_draw:
        # Proyectar a 2D (usar componentes X y Y para visualización)
        vec_2d = np.array([vec[0], -vec[1]]) * vector_scale  # Invertir Y para pantalla
        end = origin + vec_2d.astype(int)
        cv2.arrowedLine(frame, tuple(origin.astype(int)), tuple(end), color, 2, tipLength=0.15)
        cv2.putText(frame, label, tuple(end + 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Mostrar ángulos
    angle_text_y = origin[1] + size//2 + 15
    cv2.putText(frame, f"Horizontal: {angles['avg_gaze_angle_horizontal']:+.1f}°", 
               (origin_x - 45, angle_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Vertical: {angles['avg_gaze_angle_vertical']:+.1f}°", 
               (origin_x - 45, angle_text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Título
    cv2.putText(frame, "Sistema Cabeza", 
               (origin_x - 40, origin_y - size//2 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Punto de origen
    cv2.circle(frame, tuple(origin.astype(int)), 3, (255, 255, 255), -1)

def display_head_space_info(frame, head_space_vectors, angles):
    """
    Muestra información de vectores en sistema de cabeza
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    line_height = 12
    start_x = 10
    start_y = 220  # Debajo del primer panel
    
    # Calcular dimensiones del fondo
    num_lines = 8
    text_height = num_lines * line_height
    text_width = 400
    
    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (start_x - 5, start_y - 15), 
                 (start_x + text_width, start_y + text_height), 
                 (0, 0, 0), -1)
    cv2.rectangle(overlay, 
                 (start_x - 5, start_y - 15), 
                 (start_x + text_width, start_y + text_height), 
                 (100, 100, 100), 1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Título
    cv2.putText(frame, "SISTEMA CABEZA:", (start_x, start_y), 
               font, font_scale, (255, 255, 0), thickness)
    
    # Vectores de mirada en sistema cabeza
    cv2.putText(frame, f"Mirada Izq: {format_vector(head_space_vectors['left_gaze_head'])}", 
               (start_x, start_y + line_height), font, font_scale, (255, 200, 0), thickness)
    cv2.putText(frame, f"Mirada Der: {format_vector(head_space_vectors['right_gaze_head'])}", 
               (start_x, start_y + 2*line_height), font, font_scale, (255, 100, 0), thickness)
    cv2.putText(frame, f"Mirada Avg: {format_vector(head_space_vectors['avg_gaze_head'])}", 
               (start_x, start_y + 3*line_height), font, font_scale, (255, 255, 0), thickness)
    
    # Ángulos
    cv2.putText(frame, f"Ang H: {angles['avg_gaze_angle_horizontal']:+.1f}° V: {angles['avg_gaze_angle_vertical']:+.1f}°", 
               (start_x, start_y + 4*line_height), font, font_scale, (255, 255, 255), thickness)
    
    # Posiciones de ojos en sistema cabeza
    cv2.putText(frame, f"Ojo Izq Pos: {format_vector(head_space_vectors['sphere_world_l_head'])}", 
               (start_x, start_y + 5*line_height), font, font_scale, (200, 200, 255), thickness)
    cv2.putText(frame, f"Ojo Der Pos: {format_vector(head_space_vectors['sphere_world_r_head'])}", 
               (start_x, start_y + 6*line_height), font, font_scale, (200, 200, 255), thickness)
    
def toggle_pause():
    """Alterna entre pausado y reanudado"""
    global paused
    paused = not paused
    return paused

def wait_while_paused():
    """Espera mientras está en estado pausado"""
    global paused
    while paused:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):  # Presionar 'p' nuevamente para reanudar
            paused = False
        elif key == ord('q'):  # Salir incluso estando pausado
            return True
    return False

def format_vector(vector, decimals=2):
    """Formatea un vector para mostrarlo como texto"""
    return f"({vector[0]:.{decimals}f}, {vector[1]:.{decimals}f}, {vector[2]:.{decimals}f})"

def format_angle(degrees, decimals=1):
    """Formatea un ángulo en grados"""
    return f"{degrees:.{decimals}f}°"

def display_vector_info(frame, vectors_data):
    """
    Muestra la información esencial de vectores con fondo semitransparente
    """
    # Configuración de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 18
    start_x = 10
    start_y = 20
    
    # Calcular dimensiones del fondo (solo 5 líneas)
    num_lines = 5
    text_height = num_lines * line_height
    text_width = 450
    
    # Crear rectángulo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (start_x - 5, start_y - 15), 
                 (start_x + text_width, start_y + text_height), 
                 (0, 0, 0), -1)
    cv2.rectangle(overlay, 
                 (start_x - 5, start_y - 15), 
                 (start_x + text_width, start_y + text_height), 
                 (100, 100, 100), 1)
    
    # Aplicar transparencia
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 1. Dirección vector frontal (cabeza) - Coordenadas mundiales
    cv2.putText(frame, f"Frente Cabeza: {format_vector(vectors_data['avg_direction'])}", 
               (start_x, start_y), font, font_scale, (15, 255, 0), thickness)
    
    # 2. Dirección mirada promedio - Coordenadas mundiales
    cv2.putText(frame, f"Mirada Mundial: {format_vector(vectors_data['avg_combined_direction'])}", 
               (start_x, start_y + line_height), font, font_scale, (255, 255, 10), thickness)
    
    # 3. Mirada en coordenadas faciales
    cv2.putText(frame, f"Mirada Facial: {format_vector(vectors_data['gaze_facial'])}", 
               (start_x, start_y + 2*line_height), font, font_scale, (255, 200, 100), thickness)
    
    # 4. Ángulos de mirada en sistema facial
    cv2.putText(frame, f"Mirada Yaw: {format_angle(vectors_data['gaze_yaw_deg'])}", 
               (start_x, start_y + 3*line_height), font, font_scale, (255, 200, 100), thickness)
    
    # 5. Ángulos de mirada en sistema facial
    cv2.putText(frame, f"Mirada Pitch: {format_angle(vectors_data['gaze_pitch_deg'])}", 
               (start_x, start_y + 4*line_height), font, font_scale, (255, 200, 100), thickness)
    
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
                if dibujar_landmarks:
                    cv2.circle(landmarks_frame, (x, y), 3, color, -1)
                    frame[y, x] = (255, 255, 255)  # optional: also update main frame if needed

        # Highlight bounding landmarks in pink
        key_points = {}
        for name, idx in LANDMARKS.items():
            pt = landmark_to_np(face_landmarks[idx], w, h)
            key_points[name] = pt
            x, y = int(pt[0]), int(pt[1])
            if dibujar_landmarks:
                cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)

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

        # ===== DIBUJAR EJES DEL SISTEMA DE COORDENADAS FACIAL =====
        axis_length = 100  # Longitud de los ejes en píxeles

        # Eje X (Derecha/Izquierda) - ROJO
        x_end = center + right_axis * axis_length
        cv2.arrowedLine(frame, project(center), project(x_end), (0, 0, 255), 4, tipLength=0.2)
        cv2.putText(frame, "X (Derecha)", project(x_end), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Eje Y (Arriba/Abajo) - VERDE  
        y_end = center + up_axis * axis_length
        cv2.arrowedLine(frame, project(center), project(y_end), (0, 255, 0), 4, tipLength=0.2)
        cv2.putText(frame, "Y (Arriba)", project(y_end), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Eje Z (Frente) - AZUL - usando el rayo verde (forward_axis)
        z_end = center + forward_axis * axis_length
        cv2.arrowedLine(frame, project(center), project(z_end), (255, 0, 0), 4, tipLength=0.2)
        cv2.putText(frame, "Z (Frente)", project(z_end), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Punto de origen (centro de la caja facial)
        cv2.circle(frame, project(center), 6, (255, 255, 255), -1)
        cv2.putText(frame, "Origen", (project(center)[0] + 10, project(center)[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

        #Dibujamos el CUBO 

        # Colores para diferentes partes del cubo
        front_color = (255, 125, 35)    # Color original para frente y lados
        back_color = (255, 50, 200)     # Color diferente para la cara trasera (rosa/magenta)

        # Dibujar aristas del cubo con colores diferentes
        for i, j in edges:
            if (i, j) in [(4, 5), (5, 6), (6, 7), (7, 4)]:  # Cara trasera
                cv2.line(frame, cube_corners_2d[i], cube_corners_2d[j], back_color, 2)
            else:  # Caras frontal y laterales
                cv2.line(frame, cube_corners_2d[i], cube_corners_2d[j], front_color, 2)

        
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

        #print(f"Screen position: x={screen_x}, y={screen_y}")

        # Draw smoothed ray
        ray_length = 2.5 * half_depth
        ray_end = avg_origin - avg_direction * ray_length
        
        # Dibujamos el rayo frontal 
        #cv2.line(frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)
        #cv2.line(landmarks_frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)

        #===========================================================================
        # Índices de iris
        left_iris_idx = 468
        right_iris_idx = 473
        left_iris = face_landmarks[left_iris_idx]
        right_iris = face_landmarks[right_iris_idx]
        
        head_center = center  # Ya lo tienes
        R_final = np.column_stack([right_axis, up_axis, forward_axis])

        # Calcular nose_points_3d
        nose_points_3d = []
        for idx in nose_indices:
            landmark = face_landmarks[idx]
            pt = np.array([landmark.x * w, landmark.y * h, landmark.z * w])
            nose_points_3d.append(pt)
        nose_points_3d = np.array(nose_points_3d)
        
        # Posiciones 3D de iris
        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])
        
        # === CALIBRACIÓN AUTOMÁTICA EN EL PRIMER FRAME ===
        if not (left_sphere_locked and right_sphere_locked):
            time.sleep(1)

            start_center_calibration ()

            time.sleep(1)

            current_nose_scale = compute_scale(nose_points_3d)
            
            # Lock ojo izquierdo
            left_sphere_local_offset = R_final.T @ (iris_3d_left - center)
            camera_dir_world = np.array([0, 0, 1])
            camera_dir_local = R_final.T @ camera_dir_world
            left_sphere_local_offset += base_radius * camera_dir_local
            left_calibration_nose_scale = current_nose_scale
            left_sphere_locked = True
            
            # Lock ojo derecho
            right_sphere_local_offset = R_final.T @ (iris_3d_right - center)
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
        sphere_world_l = center + R_final @ scaled_offset_l
        scaled_radius_l = int(base_radius * scale_ratio_l)
        
        scale_ratio_r = current_nose_scale / right_calibration_nose_scale
        scaled_offset_r = right_sphere_local_offset * scale_ratio_r
        sphere_world_r = center + R_final @ scaled_offset_r
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

        # Mostrar ANTES de invertir Z
        print("=== ANTES de invertir Z ===")
        print(f"left_gaze_dir:  {format_vector(left_gaze_dir)}")
        print(f"right_gaze_dir: {format_vector(right_gaze_dir)}")

        # CORRECCIÓN: Invertir componente Z en ambos vectores oculares
        left_gaze_dir_original = left_gaze_dir.copy()  # Guardar copia para comparación
        right_gaze_dir_original = right_gaze_dir.copy()

        left_gaze_dir[2] = abs(left_gaze_dir[2])
        right_gaze_dir[2] = abs(right_gaze_dir[2])

        # Mostrar DESPUÉS de invertir Z
        print("=== DESPUÉS de invertir Z ===")
        print(f"left_gaze_dir:  {format_vector(left_gaze_dir)}")
        print(f"right_gaze_dir: {format_vector(right_gaze_dir)}")

        # Mostrar CAMBIO específico
        print("=== CAMBIO en Z ===")
        print(f"Left Z: {left_gaze_dir_original[2]:.3f} -> {left_gaze_dir[2]:.3f}")
        print(f"Right Z: {right_gaze_dir_original[2]:.3f} -> {right_gaze_dir[2]:.3f}")

        
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
        

        # ===== CONVERTIR DIRECCIÓN DE MIRADA A COORDENADAS FACIALES =====

        # Convertir dirección promedio de mirada al sistema facial
        gaze_facial = convert_gaze_to_facial_coordinates(
            avg_combined_direction, center, right_axis, up_axis, forward_axis
        )

        # Convertir a ángulos
        gaze_yaw_deg, gaze_pitch_deg = gaze_to_angles_facial_coordinates(gaze_facial)

        # También convertir direcciones individuales de los ojos
        left_gaze_facial = convert_gaze_to_facial_coordinates(
            left_gaze_dir, center, right_axis, up_axis, forward_axis
        )
        right_gaze_facial = convert_gaze_to_facial_coordinates(
            right_gaze_dir, center, right_axis, up_axis, forward_axis
        )

        # Calcular ángulos individuales
        left_yaw_deg, left_pitch_deg = gaze_to_angles_facial_coordinates(left_gaze_facial)
        right_yaw_deg, right_pitch_deg = gaze_to_angles_facial_coordinates(right_gaze_facial)


        # ===== DIBUJAR VECTOR DE MIRADA EN EL SISTEMA DE COORDENADAS =====
        # Dibujar vector de mirada promedio en el sistema facial (desde el centro) BIEN!
        gaze_visual_length = 100

        # CORRECCIÓN: Convertir el vector de mirada de vuelta a coordenadas mundiales para dibujarlo
        gaze_world_visual = (right_axis * gaze_facial[0] + 
                            up_axis * gaze_facial[1] + 
                            forward_axis * gaze_facial[2])
        gaze_world_visual /= np.linalg.norm(gaze_world_visual)  # Normalizar

        gaze_end_visual = center + gaze_world_visual * gaze_visual_length

        # Dibujar línea de mirada en el sistema de coordenadas
        cv2.arrowedLine(frame, project(center), project(gaze_end_visual), 
                        (255, 255, 0), 3, tipLength=0.2)
        cv2.putText(frame, "Mirada", project(gaze_end_visual), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # ===== AGREGAR TEXTO CON LOS VALORES DE LOS VECTORES =====
        
        # Preparar datos para la función
        # Actualizar el diccionario vectors_data con solo lo esencial
        vectors_data = {
            'avg_combined_direction': avg_combined_direction,
            'avg_direction': avg_direction,
            'gaze_facial': gaze_facial,
            'gaze_yaw_deg': gaze_yaw_deg,
            'gaze_pitch_deg': gaze_pitch_deg
        }
        
        print(f"Frente Cabeza: {format_vector(vectors_data['avg_direction'])}")
        print(f"Mirada Mundial: {format_vector(vectors_data['avg_combined_direction'])}")
        print(f"Mirada Facial: {format_vector(vectors_data['gaze_facial'])}")
        # Llamar a la función
        display_vector_info(frame, vectors_data)
        

        # Mostrar estado de pausa en la pantalla
        if paused:
            cv2.putText(frame, "PAUSADO - Presiona 'P' para continuar", 
                       (MONITOR_WIDTH // 2 - 200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Eye Tracking - Gaze Rays", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        dibujar_landmarks = False

    elif key == ord('c'):
        calibration_offset_yaw = 180 - raw_yaw_deg
        calibration_offset_pitch = 180 - raw_pitch_deg
        print(f"[Calibrated] Offset Yaw: {calibration_offset_yaw}, Offset Pitch: {calibration_offset_pitch}")

    elif key == ord('s'):
        dibujar_landmarks = True

    elif key == ord('r'):
        #aqui iniciariamos la recoleccion de datos para posteriormente usarlos con el fin de entrenar un modelo que me permita mapear las coordenadas a la pantalla
        show_multipoint ()

        #recolectamos las direcciones de los vectores de mirada, del vector promedio de la mirada y del vector del frente, asi como otros datos de interes mientras se muestran los 5 puntos

        #Guardamos esos datos donde se enlaza zonas de la pantalla con ciertas caracteristicas o vectores en un pkl
    
    elif key == ord('p'):  # Tecla P para pausar/reanudar
        toggle_pause()
        if paused:
            cv2.putText(frame, "PAUSADO - Presiona 'P' para continuar", 
                       (MONITOR_WIDTH // 2 - 200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Eye Tracking - Gaze Rays", frame)

    if paused:
        if wait_while_paused():
            break  # Salir si se presionó 'q' durante la pausa

cap.release()
cv2.destroyAllWindows()