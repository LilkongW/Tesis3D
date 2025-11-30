import pygame
import cv2
import numpy as np
import pandas as pd
import os
import threading
import queue
from datetime import datetime
from sklearn.cluster import KMeans
# Importar las utilidades de eye tracking
from eye_tracker_utils_lite import process_frame, reset_tracker_state

# ==========================================
# CONSTANTES GLOBALES
# ==========================================
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
ROWS, COLS = 3, 3
CELL_WIDTH = SCREEN_WIDTH // COLS
CELL_HEIGHT = SCREEN_HEIGHT // ROWS

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0, 100)  # Con transparencia
GRAY = (50, 50, 50)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (100, 160, 210)

# Rutas
CALIBRATION_MATRIX_PATH = 'calibracion_matriz_h.npy'
CALIBRATION_DATA_PATH = 'calibracion_datos.csv'

# ==========================================
# FUNCIONES MATEMÁTICAS
# ==========================================
def project_vector_to_plane(vectors_3d):
    """Convierte vectores 3D (x, y, z) en puntos 2D proyectados."""
    z = vectors_3d[:, 2].copy()
    z[z == 0] = 1e-6
    x_proj = vectors_3d[:, 0] / np.abs(z)
    y_proj = vectors_3d[:, 1] / np.abs(z)
    return np.column_stack((x_proj, y_proj))

def find_homography_manual(src_pts, dst_pts):
    """Calcula la matriz de homografía H."""
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    U, S, Vh = np.linalg.svd(np.array(A))
    return Vh[-1, :].reshape(3, 3)

def perspective_transform_manual(points, H):
    """Aplica transformación de perspectiva."""
    points = np.array(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    transformed_hom = np.dot(points_hom, H.T)
    w = transformed_hom[:, 2:]
    w[w == 0] = 1e-10
    return transformed_hom[:, :2] / w

def sort_grid_points(points):
    """Ordena 9 puntos en cuadrícula de arriba-izq a abajo-der."""
    points = points[points[:, 1].argsort()]
    row1 = points[0:3][points[0:3][:, 0].argsort()]
    row2 = points[3:6][points[3:6][:, 0].argsort()]
    row3 = points[6:9][points[6:9][:, 0].argsort()]
    return np.vstack((row1, row2, row3))

def get_cell_center(row, col):
    """Calcula el centro de una celda en la cuadrícula."""
    x = col * CELL_WIDTH + CELL_WIDTH // 2
    y = row * CELL_HEIGHT + CELL_HEIGHT // 2
    return (x, y)

# ==========================================
# CLASE DETECTOR DE PARPADEOS
# ==========================================
class BlinkDetector:
    def __init__(self, frames_threshold=30):
        self.frames_threshold = frames_threshold
        self.frames_without_detection = 0
        self.blink_detected = False
        self.last_valid_cell = None
    
    def update(self, detection_valid, current_cell):
        """
        Actualiza el detector con el estado actual.
        Retorna True si se detectó un parpadeo completo.
        """
        if not detection_valid:
            # No hay detección
            self.frames_without_detection += 1
            
            # Si alcanzamos el umbral, marcamos que empezó el parpadeo
            if self.frames_without_detection >= self.frames_threshold:
                if not self.blink_detected and self.last_valid_cell is not None:
                    self.blink_detected = True
        else:
            # Hay detección válida
            if self.blink_detected:
                # Parpadeo completado: hubo pérdida y ahora recuperamos
                self.blink_detected = False
                self.frames_without_detection = 0
                # Retornar la celda donde se hizo el parpadeo
                blink_cell = self.last_valid_cell
                self.last_valid_cell = current_cell
                return blink_cell
            else:
                # Detección normal, resetear contador
                self.frames_without_detection = 0
                self.last_valid_cell = current_cell
        
        return None
    
    def reset(self):
        """Reinicia el detector."""
        self.frames_without_detection = 0
        self.blink_detected = False
        self.last_valid_cell = None

# ==========================================
# CLASE PARA CAPTURA DE VIDEO EN HILO
# ==========================================
class VideoCapture:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.queue = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
    
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                if not self.queue.full():
                    try:
                        self.queue.put(frame, block=False)
                    except queue.Full:
                        pass
    
    def read(self):
        if not self.queue.empty():
            return self.queue.get()
        return None
    
    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# ==========================================
# CLASE PRINCIPAL DE LA APLICACIÓN
# ==========================================
class EyeTrackingApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Sistema Eye Tracking")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        self.state = "MENU"  # MENU, CALIBRATION, TESTING
        self.running = True
        
        # Datos de calibración
        self.calibration_data = []
        self.H_matrix = None
        self.video_capture = None
        
        # Posiciones de calibración (9 puntos)
        self.calibration_positions = []
        for r in range(ROWS):
            for c in range(COLS):
                self.calibration_positions.append(get_cell_center(r, c))
        
        self.current_fixation_idx = 0
        self.fixation_start_time = 0
        self.fixation_duration = 2000  # 2 segundos por punto
        
        # Para modo de prueba
        self.current_gaze_cell = None
        self.selected_cells = {}  # {(row, col): 'green' o 'red'}
        self.blink_detector = BlinkDetector()
        
    def draw_button(self, text, rect, mouse_pos):
        """Dibuja un botón con efecto hover."""
        is_hover = rect.collidepoint(mouse_pos)
        color = BUTTON_HOVER if is_hover else BUTTON_COLOR
        
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, rect, 3, border_radius=10)
        
        text_surf = self.font_medium.render(text, True, WHITE)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)
        
        return is_hover
    
    def draw_menu(self):
        """Dibuja el menú principal."""
        self.screen.fill(BLACK)
        
        # Título
        title = self.font_large.render("Sistema Eye Tracking", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 200))
        self.screen.blit(title, title_rect)
        
        # Botones
        mouse_pos = pygame.mouse.get_pos()
        
        calibrate_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, 400, 400, 80)
        test_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, 520, 400, 80)
        exit_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, 640, 400, 80)
        
        calibrate_hover = self.draw_button("Calibrar", calibrate_rect, mouse_pos)
        test_hover = self.draw_button("Probar Calibración", test_rect, mouse_pos)
        exit_hover = self.draw_button("Salir", exit_rect, mouse_pos)
        
        # Info sobre calibración existente
        if os.path.exists(CALIBRATION_MATRIX_PATH):
            info_text = self.font_small.render("✓ Calibración cargada", True, GREEN)
            info_rect = info_text.get_rect(center=(SCREEN_WIDTH // 2, 800))
            self.screen.blit(info_text, info_rect)
        
        return calibrate_hover, test_hover, exit_hover, calibrate_rect, test_rect, exit_rect
    
    def draw_calibration_grid(self):
        """Dibuja la cuadrícula de calibración."""
        # Dibujar líneas de cuadrícula
        for i in range(1, ROWS):
            pygame.draw.line(self.screen, GRAY, 
                           (0, i * CELL_HEIGHT), 
                           (SCREEN_WIDTH, i * CELL_HEIGHT), 2)
        for j in range(1, COLS):
            pygame.draw.line(self.screen, GRAY, 
                           (j * CELL_WIDTH, 0), 
                           (j * CELL_WIDTH, SCREEN_HEIGHT), 2)
        
        # Dibujar punto de fijación actual
        if self.current_fixation_idx < len(self.calibration_positions):
            pos = self.calibration_positions[self.current_fixation_idx]
            pygame.draw.circle(self.screen, RED, pos, 30)
            
            # Mostrar progreso
            progress_text = f"Punto {self.current_fixation_idx + 1} / {len(self.calibration_positions)}"
            text_surf = self.font_small.render(progress_text, True, WHITE)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, 50))
            self.screen.blit(text_surf, text_rect)
    
    def draw_testing_grid(self):
        """Dibuja la cuadrícula para pruebas."""
        # Dibujar celdas con sus colores
        for row in range(ROWS):
            for col in range(COLS):
                x = col * CELL_WIDTH
                y = row * CELL_HEIGHT
                cell_key = (row, col)
                
                # Verificar si la celda está seleccionada
                if cell_key in self.selected_cells:
                    color = self.selected_cells[cell_key]
                    if color == 'green':
                        # Verde para hover
                        s = pygame.Surface((CELL_WIDTH, CELL_HEIGHT))
                        s.set_alpha(100)
                        s.fill((0, 255, 0))
                        self.screen.blit(s, (x, y))
                        pygame.draw.rect(self.screen, (0, 255, 0), 
                                       (x, y, CELL_WIDTH, CELL_HEIGHT), 5)
                    elif color == 'red':
                        # Rojo para selección por parpadeo
                        s = pygame.Surface((CELL_WIDTH, CELL_HEIGHT))
                        s.set_alpha(150)
                        s.fill((255, 0, 0))
                        self.screen.blit(s, (x, y))
                        pygame.draw.rect(self.screen, (255, 0, 0), 
                                       (x, y, CELL_WIDTH, CELL_HEIGHT), 8)
                        
                        # Mostrar ícono de checkmark
                        center_x = x + CELL_WIDTH // 2
                        center_y = y + CELL_HEIGHT // 2
                        check_text = "✓"
                        text_surf = self.font_large.render(check_text, True, WHITE)
                        text_rect = text_surf.get_rect(center=(center_x, center_y))
                        self.screen.blit(text_surf, text_rect)
        
        # Dibujar líneas de cuadrícula encima
        for i in range(1, ROWS):
            pygame.draw.line(self.screen, WHITE, 
                           (0, i * CELL_HEIGHT), 
                           (SCREEN_WIDTH, i * CELL_HEIGHT), 3)
        for j in range(1, COLS):
            pygame.draw.line(self.screen, WHITE, 
                           (j * CELL_WIDTH, 0), 
                           (j * CELL_WIDTH, SCREEN_HEIGHT), 3)
        
        # Resaltar celda actual donde se está mirando (solo si no está seleccionada en rojo)
        if self.current_gaze_cell is not None:
            row, col = self.current_gaze_cell
            cell_key = (row, col)
            
            # Solo mostrar verde si no está seleccionada en rojo
            if cell_key not in self.selected_cells or self.selected_cells[cell_key] != 'red':
                x = col * CELL_WIDTH
                y = row * CELL_HEIGHT
                
                # Verde semi-transparente para hover
                s = pygame.Surface((CELL_WIDTH, CELL_HEIGHT))
                s.set_alpha(100)
                s.fill((0, 255, 0))
                self.screen.blit(s, (x, y))
                
                # Borde verde
                pygame.draw.rect(self.screen, (0, 255, 0), 
                               (x, y, CELL_WIDTH, CELL_HEIGHT), 5)
        
        # Mostrar instrucciones
        inst_text = "Mira una zona y parpadea para seleccionar - ESC para volver - R para reiniciar"
        text_surf = self.font_small.render(inst_text, True, WHITE)
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, 50))
        self.screen.blit(text_surf, text_rect)
        
        # Mostrar contador de selecciones
        selected_count = sum(1 for v in self.selected_cells.values() if v == 'red')
        count_text = f"Seleccionadas: {selected_count}/9"
        count_surf = self.font_small.render(count_text, True, WHITE)
        count_rect = count_surf.get_rect(topright=(SCREEN_WIDTH - 20, 100))
        self.screen.blit(count_surf, count_rect)
    
    def start_calibration(self):
        """Inicia el proceso de calibración."""
        self.state = "CALIBRATION"
        self.calibration_data = []
        self.current_fixation_idx = 0
        self.fixation_start_time = pygame.time.get_ticks()
        
        # Reset del estado del tracker
        reset_tracker_state()
        
        # Iniciar captura de video
        self.video_capture = VideoCapture(0)
        print("Calibración iniciada - Procesando video...")
    
    def process_calibration_frame(self):
        """Procesa un frame durante la calibración."""
        frame = self.video_capture.read()
        if frame is None:
            return
        
        try:
            # Procesar frame con eye tracker
            data = process_frame(frame)
            
            # Si la detección es válida, guardar datos
            if data.get('valid_deteccion', False):
                gaze_data = {
                    'gaze_x': data.get('gaze_x'),
                    'gaze_y': data.get('gaze_y'),
                    'gaze_z': data.get('gaze_z'),
                    'fixation_point': self.current_fixation_idx,
                    'timestamp': pygame.time.get_ticks()
                }
                self.calibration_data.append(gaze_data)
        except Exception as e:
            print(f"Error procesando frame: {e}")
    
    def finish_calibration(self):
        """Finaliza la calibración y calcula la matriz H."""
        if self.video_capture:
            self.video_capture.stop()
            self.video_capture = None
        
        cv2.destroyAllWindows()
        
        if len(self.calibration_data) < 50:
            print("⚠️ Datos insuficientes para calibración")
            self.state = "MENU"
            return
        
        # Guardar datos crudos
        df = pd.DataFrame(self.calibration_data)
        df.to_csv(CALIBRATION_DATA_PATH, index=False)
        print(f"✓ Datos guardados en {CALIBRATION_DATA_PATH}")
        
        # Calcular matriz de calibración
        try:
            vectors_3d = df[['gaze_x', 'gaze_y', 'gaze_z']].values.astype(np.float32)
            points_2d_projected = project_vector_to_plane(vectors_3d)
            
            # K-Means para encontrar los 9 centros
            kmeans = KMeans(n_clusters=9, random_state=42, n_init='auto')
            kmeans.fit(points_2d_projected)
            centers_projected = kmeans.cluster_centers_
            
            # Targets teóricos (pantalla normalizada)
            target_vals = [0.2, 0.5, 0.8]
            dst_points_norm = np.array([[x, y] for y in target_vals for x in target_vals], 
                                      dtype=np.float32)
            
            # Ordenar puntos
            src_sorted = sort_grid_points(centers_projected)
            dst_sorted = sort_grid_points(dst_points_norm)
            
            # Calcular matriz H
            self.H_matrix = find_homography_manual(src_sorted, dst_sorted)
            
            # Guardar matriz
            np.save(CALIBRATION_MATRIX_PATH, self.H_matrix)
            
            # Calcular error
            src_px = perspective_transform_manual(src_sorted, self.H_matrix) * [SCREEN_WIDTH, SCREEN_HEIGHT]
            dst_px = dst_sorted * [SCREEN_WIDTH, SCREEN_HEIGHT]
            errors = np.linalg.norm(src_px - dst_px, axis=1)
            avg_error = np.mean(errors)
            
            print(f"✅ Calibración completada - Error promedio: {avg_error:.2f} px")
            
        except Exception as e:
            print(f"❌ Error en calibración: {e}")
        
        self.state = "MENU"
    
    def start_testing(self):
        """Inicia el modo de prueba."""
        # Cargar matriz de calibración
        if not os.path.exists(CALIBRATION_MATRIX_PATH):
            print("⚠️ No hay calibración disponible")
            return
        
        self.H_matrix = np.load(CALIBRATION_MATRIX_PATH)
        self.state = "TESTING"
        
        # Reset del estado del tracker y detector de parpadeo
        reset_tracker_state()
        self.blink_detector.reset()
        self.selected_cells = {}
        self.current_gaze_cell = None
        
        self.video_capture = VideoCapture(0)
        print("Modo de prueba iniciado")
    
    def process_testing_frame(self):
        """Procesa un frame durante las pruebas."""
        frame = self.video_capture.read()
        if frame is None:
            return
        
        try:
            # Procesar frame
            data = process_frame(frame)
            
            detection_valid = data.get('valid_deteccion', False)
            
            if detection_valid:
                # Obtener vector de mirada
                gaze_vector = np.array([
                    data.get('gaze_x'),
                    data.get('gaze_y'),
                    data.get('gaze_z')
                ]).reshape(1, -1)
                
                # Proyectar a 2D
                gaze_2d = project_vector_to_plane(gaze_vector)
                
                # Aplicar calibración
                gaze_calibrated = perspective_transform_manual(gaze_2d, self.H_matrix)
                
                # Convertir a píxeles
                gaze_x_px = gaze_calibrated[0, 0] * SCREEN_WIDTH
                gaze_y_px = gaze_calibrated[0, 1] * SCREEN_HEIGHT
                
                # Determinar en qué celda está mirando
                if 0 <= gaze_x_px < SCREEN_WIDTH and 0 <= gaze_y_px < SCREEN_HEIGHT:
                    col = int(gaze_x_px // CELL_WIDTH)
                    row = int(gaze_y_px // CELL_HEIGHT)
                    self.current_gaze_cell = (row, col)
                else:
                    self.current_gaze_cell = None
            else:
                self.current_gaze_cell = None
            
            # Actualizar detector de parpadeo
            blink_cell = self.blink_detector.update(detection_valid, self.current_gaze_cell)
            
            # Si se detectó un parpadeo en una celda, marcarla en rojo
            if blink_cell is not None:
                self.selected_cells[blink_cell] = 'red'
                print(f"✓ Celda seleccionada por parpadeo: {blink_cell}")
                
        except Exception as e:
            print(f"Error procesando frame: {e}")
            self.current_gaze_cell = None
    
    def stop_testing(self):
        """Detiene el modo de prueba."""
        if self.video_capture:
            self.video_capture.stop()
            self.video_capture = None
        cv2.destroyAllWindows()
        self.state = "MENU"
    
    def run(self):
        """Bucle principal de la aplicación."""
        while self.running:
            # Manejo de eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state == "CALIBRATION":
                            self.finish_calibration()
                        elif self.state == "TESTING":
                            self.stop_testing()
                        elif self.state == "MENU":
                            self.running = False
                    
                    # Tecla R para reiniciar selecciones en modo testing
                    if event.key == pygame.K_r and self.state == "TESTING":
                        self.selected_cells = {}
                        self.blink_detector.reset()
                        print("✓ Selecciones reiniciadas")
                
                if event.type == pygame.MOUSEBUTTONDOWN and self.state == "MENU":
                    mouse_pos = pygame.mouse.get_pos()
                    _, _, _, cal_rect, test_rect, exit_rect = self.draw_menu()
                    
                    if cal_rect.collidepoint(mouse_pos):
                        self.start_calibration()
                    elif test_rect.collidepoint(mouse_pos):
                        self.start_testing()
                    elif exit_rect.collidepoint(mouse_pos):
                        self.running = False
            
            # Renderizado según el estado
            self.screen.fill(BLACK)
            
            if self.state == "MENU":
                self.draw_menu()
            
            elif self.state == "CALIBRATION":
                self.draw_calibration_grid()
                
                # Procesar frame de video
                self.process_calibration_frame()
                
                # Verificar si se completó el tiempo de fijación
                elapsed = pygame.time.get_ticks() - self.fixation_start_time
                if elapsed >= self.fixation_duration:
                    self.current_fixation_idx += 1
                    self.fixation_start_time = pygame.time.get_ticks()
                    
                    # Si terminamos todos los puntos
                    if self.current_fixation_idx >= len(self.calibration_positions):
                        self.finish_calibration()
            
            elif self.state == "TESTING":
                self.draw_testing_grid()
                self.process_testing_frame()
            
            pygame.display.flip()
            self.clock.tick(30)
        
        # Limpieza
        if self.video_capture:
            self.video_capture.stop()
        cv2.destroyAllWindows()
        pygame.quit()

# ==========================================
# PUNTO DE ENTRADA
# ==========================================
if __name__ == "__main__":
    app = EyeTrackingApp()
    app.run()