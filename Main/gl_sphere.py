# gl_sphere.py (Modificado: Ajustes finales de anillo e iris)

import sys
import numpy as np
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.error import GLError

# Singleton references
app = None
window = None
sphere_widget = None

# Buffer de imagen global con candado de seguridad
gl_image_buffer = None 
buffer_lock = threading.Lock() 

class DataSignal(QObject):
    # Añadido 'float' para el radio de la pupila
    update_data = pyqtSignal(int, int, int, int, int, int, float)

data_signal = DataSignal()

class SphereWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.sphere_rot_x = 0
        self.sphere_rot_y = 0
        
        self.sphere_vertices, self.sphere_indices = self.generate_wireframe_sphere(30, 30)
        
        self.center_point_quadric = gluNewQuadric()
        
        self.pupil_scaled_radius = 0.2 # Radio por defecto (en unidades GL)
        
        # --- ¡MODIFICADO! ---
        self.iris_fixed_border = 0.12  # Reducido de 0.15 a 0.12
        # --- FIN MODIFICACIÓN ---
        
        self.draw_gaze_vector = False # Flag para dibujar la línea

        self.camera_position = np.array([0.0, 0.0, -3.0])
        self.ray_origin = None
        self.ray_direction = None

        self.sphere_center_x = 320
        self.sphere_center_y = 240
        
        data_signal.update_data.connect(self.receive_cv_data)

    def receive_cv_data(self, pupil_x, pupil_y, center_x, center_y, width, height, scaled_radius):
        """Ranura llamada desde el hilo principal de CV."""
        
        self.pupil_scaled_radius = scaled_radius
        self.sphere_center_x = center_x
        self.sphere_center_y = center_y

        if pupil_x > 0:
            self.draw_gaze_vector = True # Activar dibujo de línea
            self.compute_and_apply_rotation(pupil_x, pupil_y, center_x, center_y, width, height)
        else:
            self.draw_gaze_vector = False # Desactivar dibujo de línea
        
        self.update()


    def compute_and_apply_rotation(self, x, y, center_x, center_y, screen_width, screen_height):
        """Contiene la lógica pesada de cálculo de rotación."""
        
        # ... (TODA LA LÓGICA DE CÁLCULO DE ROTACIÓN SE MANTIENE EXACTAMENTE IGUAL) ...
        viewport_width = screen_width
        viewport_height = screen_height
        fov_y_deg = 45.0
        aspect_ratio = viewport_width / viewport_height
        far_clip = 100.0
        camera_position = np.array([0.0, 0.0, 3.0])
        fov_y_rad = np.radians(fov_y_deg)
        half_height_far = np.tan(fov_y_rad / 2) * far_clip
        half_width_far = half_height_far * aspect_ratio
        ndc_x = (2.0 * x) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * y) / viewport_height
        far_x = ndc_x * half_width_far
        far_y = ndc_y * half_height_far
        far_z = camera_position[2] - far_clip
        far_point = np.array([far_x, far_y, far_z])
        ray_origin = camera_position
        ray_direction = far_point - camera_position
        ray_direction /= np.linalg.norm(ray_direction)
        ray_direction = -ray_direction 
        self.ray_origin = ray_origin
        self.ray_direction = ray_direction
        inner_radius = 1.0 / 1.05
        sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
        sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
        sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])
        origin = ray_origin
        direction = -ray_direction
        L = origin - sphere_center
        a = np.dot(direction, direction)
        b = 2 * np.dot(direction, L)
        c = np.dot(L, L) - inner_radius**2
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        t = None
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        if t is None:
            return
        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        target_direction = intersection_local / np.linalg.norm(intersection_local)
        
        circle_local_center = np.array([0.0, 0.0, inner_radius])
        circle_local_center /= np.linalg.norm(circle_local_center)
        rotation_axis = np.cross(circle_local_center, target_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        if rotation_axis_norm < 1e-6:
            return
        rotation_axis /= rotation_axis_norm
        dot = np.dot(circle_local_center, target_direction)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        t_ = 1 - c
        x_, y_, z_ = rotation_axis
        rotation_matrix = np.array([
            [t_*x_*x_ + c, t_*x_*y_ - s*z_, t_*x_*z_ + s*y_],
            [t_*x_*y_ + s*z_, t_*y_*y_ + c, t_*y_*z_ - s*x_],
            [t_*x_*z_ - s*y_, t_*y_*z_ + s*x_, t_*z_*z_ + c]
        ])

        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        if sy < 1e-6:
            x_rot = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y_rot = np.arctan2(-rotation_matrix[2, 0], sy)
        else:
            x_rot = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y_rot = np.arctan2(-rotation_matrix[2, 0], sy)

        self.sphere_rot_x = np.degrees(x_rot)
        self.sphere_rot_y = np.degrees(y_rot)


    def generate_wireframe_sphere(self, lat_div, lon_div):
        vertices = []
        indices = []
        for i in range(lat_div + 1):
            lat = np.pi * (-0.5 + float(i) / lat_div)
            z = np.sin(lat)
            zr = np.cos(lat)
            for j in range(lon_div + 1):
                lon = 2 * np.pi * float(j) / lon_div
                x = np.cos(lon) * zr
                y = np.sin(lon) * zr
                vertices.append((x, y, z))
        for i in range(lat_div):
            for j in range(lon_div):
                p1 = i * (lon_div + 1) + j
                p2 = p1 + lon_div + 1
                indices.append((p1, p2))
                indices.append((p1, p1 + 1))
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)
    
    def generate_solid_circle(self, radius, z_offset, segments=32):
        vertices = []
        vertices.append((0.0, 0.0, z_offset)) 
        for i in range(segments + 1): 
            angle = 2.0 * np.pi * i / segments
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            vertices.append((x, y, z_offset))
        return np.array(vertices, dtype=np.float32)

    def generate_circle_loop(self, radius, z_offset, segments=32):
        """Genera vértices para un anillo (usando GL_LINE_LOOP)."""
        vertices = []
        for i in range(segments): # No necesita +1 para GL_LINE_LOOP
            angle = 2.0 * np.pi * i / segments
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            vertices.append((x, y, z_offset))
        return np.array(vertices, dtype=np.float32)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) 
        glClearColor(0.1, 0.1, 0.1, 1.0) 

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / max(1, h), 0.1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """Método de renderizado de OpenGL (ejecutado en el hilo correcto)."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glTranslatef(0.0, 0.0, -3)

        viewport_width = self.width()
        viewport_height = self.height()

        gl_x = (self.sphere_center_x / viewport_width) * 2.0 - 1.0
        gl_y = 1.0 - (self.sphere_center_y / viewport_height) * 2.0

        glTranslatef(gl_x * 1.5, gl_y * 1.5, 0.0)
        
        glPushMatrix()
        # Aplicar la rotación de la mirada
        glRotatef(self.sphere_rot_x, 1, 0, 0)
        glRotatef(self.sphere_rot_y, 0, 1, 0)
        
        # Dibujar el punto central
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL) 
        glDisable(GL_DEPTH_TEST) 
        glColor3f(0.0, 1.0, 1.0) # Color CYAN
        gluSphere(self.center_point_quadric, 0.03, 16, 16) # Radio 0.03
        glEnable(GL_DEPTH_TEST) 

        # 1. Dibujar Esclera (la malla blanca)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) 
        glColor3f(1.0, 1.0, 1.0) 
        glLineWidth(1.0) 
        glBegin(GL_LINES)
        for i1, i2 in self.sphere_indices:
            glVertex3fv(self.sphere_vertices[i1])
            glVertex3fv(self.sphere_vertices[i2])
        glEnd()
        
        # 2. Dibujar Iris y Pupila (sólidos)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL) 
        glDisable(GL_DEPTH_TEST) 
        
        # 2a. Dibujar Iris (gris claro)
        glColor3f(0.7, 0.7, 0.7) # Gris claro
        iris_radius = self.pupil_scaled_radius + self.iris_fixed_border
        iris_radius = max(iris_radius, self.pupil_scaled_radius + 0.01)
        iris_vertices = self.generate_solid_circle(iris_radius, z_offset=1.001, segments=32)
        glBegin(GL_TRIANGLE_FAN)
        for vertex in iris_vertices:
            glVertex3fv(vertex)
        glEnd()

        # 2b. Dibujar Pupila (negra)
        glColor3f(0.0, 0.0, 0.0) 
        pupil_vertices = self.generate_solid_circle(self.pupil_scaled_radius, z_offset=1.002, segments=32)
        glBegin(GL_TRIANGLE_FAN)
        for vertex in pupil_vertices:
            glVertex3fv(vertex)
        glEnd()

        # --- ¡MODIFICADO! Dibujar el ANILLO amarillo dentro de la pupila ---
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) # <-- Cambiar a modo LÍNEA
        glLineWidth(2.5) # <-- Grosor aumentado
        glColor3f(1.0, 1.0, 0.0) # Color amarillo
        
        # Radio al 95% del borde de la pupila
        yellow_ring_radius = self.pupil_scaled_radius * 1.0
        
        yellow_ring_vertices = self.generate_circle_loop(yellow_ring_radius, z_offset=1.003, segments=32)
        
        glBegin(GL_LINE_LOOP) # <-- Usar GL_LINE_LOOP para un anillo
        for vertex in yellow_ring_vertices:
            glVertex3fv(vertex)
        glEnd()
        # --- FIN MODIFICACIÓN ---

        glEnable(GL_DEPTH_TEST) 
        
        # --- BLOQUE DEL VECTOR DE MIRADA ---
        if self.draw_gaze_vector: 
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) 
            glDisable(GL_DEPTH_TEST) 
            
            # Parte Interna (Color 50, 150, 255)
            glLineWidth(2.5) # Grosor menor
            glColor3f(0.196, 0.588, 1.0) # Color (50, 150, 255)
            glBegin(GL_LINES)
            glVertex3f(0.0, 0.0, 0.0) # Origen en el centro
            glVertex3f(0.0, 0.0, 1.0) # Destino en la superficie
            glEnd()
            
            # Parte Externa (Cyan)
            glLineWidth(4.0) # Grosor mayor
            glColor3f(0.0, 1.0, 1.0) # Cyan brillante
            glBegin(GL_LINES)
            glVertex3f(0.0, 0.0, 1.0) # Origen en la superficie
            glVertex3f(0.0, 0.0, 1.5) # Destino final extendido
            glEnd()
            
            glEnable(GL_DEPTH_TEST) 
        # --- FIN BLOQUE ---

        glPopMatrix()
        
        self.grab_frame_to_buffer()

    def grab_frame_to_buffer(self):
        """
        Captura el framebuffer actual usando QOpenGLWidget.grabFramebuffer()
        y lo guarda en el buffer global de forma segura.
        """
        global gl_image_buffer, buffer_lock
        
        q_image = self.grabFramebuffer()
        
        if q_image.isNull():
            return

        q_image = q_image.convertToFormat(QImage.Format_RGB888)
        
        width = q_image.width()
        height = q_image.height()
        
        ptr = q_image.bits()
        ptr.setsize(height * width * 3) 
        
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3)).copy()

        with buffer_lock:
            gl_image_buffer = arr 


def start_gl_window():
    global app, window, sphere_widget
    app = QApplication(sys.argv) 
    window = QMainWindow()
    sphere_widget = SphereWidget()
    window.setCentralWidget(sphere_widget)
    window.setWindowTitle("Ojo Virtual - Modelo 3D")
    window.resize(640, 480)
    
    window.show() 
    
    return app

def get_latest_rendered_image():
    """Llamado desde el hilo CV para obtener la imagen del buffer."""
    global gl_image_buffer, buffer_lock
    image_copy = None
    with buffer_lock: 
        if gl_image_buffer is not None:
            image_copy = gl_image_buffer.copy()
    return image_copy 

def update_sphere_rotation_signal(x, y, center_x, center_y, screen_width, screen_height, scaled_radius):
    """
    Función llamada desde el hilo principal de CV. 
    Solo EMITE la señal, no realiza operaciones OpenGL.
    """
    global data_signal
    if data_signal is not None:
        data_signal.update_data.emit(x, y, center_x, center_y, screen_width, screen_height, scaled_radius)