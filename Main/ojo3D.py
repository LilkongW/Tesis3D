import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os

class EyeTrackingVisualizer:
    """Clase para visualizar el movimiento del ojo en 3D desde datos CSV."""
    
    def __init__(self, csv_file, blend_file=None):
        self.csv_file = csv_file
        self.blend_file = blend_file
        self.data = None
        self.fig = None
        self.ax = None
        self.current_frame = 0
        
        # Mesh del ojo desde Blender
        self.eye_mesh_vertices = None
        self.eye_mesh_faces = None
        self.eye_mesh_collection = None
        
        # Elementos gráficos
        self.eye_sphere = None
        self.pupil_poly = None # Añadido
        self.gaze_line = None
        self.gaze_point = None
        self.pupil_ellipse = None
        self.trail_line = None
        self.trail_points = []
        self.max_trail_length = 50
        
    def load_3d_model(self):
        """Carga el modelo del ojo desde varios formatos 3D."""
        if not self.blend_file or not os.path.exists(self.blend_file):
            print("No se proporcionó archivo 3D o no existe. Usando esfera por defecto.")
            return False
        
        file_ext = os.path.splitext(self.blend_file)[1].lower()
        
        # Intentar cargar con trimesh (soporta OBJ, STL, PLY, GLB, etc.)
        try:
            import trimesh
            
            print(f"Cargando modelo 3D: {self.blend_file}")
            mesh = trimesh.load(self.blend_file)
            
            # Si es una escena con múltiples objetos, tomar el primero
            if isinstance(mesh, trimesh.Scene):
                print("Escena detectada, extrayendo geometría...")
                geometries = list(mesh.geometry.values())
                if len(geometries) == 0:
                    print("No se encontró geometría en la escena")
                    return False
                mesh = geometries[0]
            
            # Extraer vértices y caras
            self.eye_mesh_vertices = np.array(mesh.vertices)
            self.eye_mesh_faces = np.array(mesh.faces)
            
            print(f"✅ Modelo cargado: {len(self.eye_mesh_vertices)} vértices, {len(self.eye_mesh_faces)} caras")
            return True
            
        except ImportError:
            print("\n❌ ERROR: trimesh no está instalado.")
            print("Para cargar modelos 3D, instala trimesh:")
            print("  pip install trimesh")
            print("\nFormatos soportados con trimesh:")
            print("  • OBJ (.obj)")
            print("  • STL (.stl)")
            print("  • PLY (.ply)")
            print("  • GLB/GLTF (.glb, .gltf)")
            print("  • OFF (.off)")
            print("  • Y muchos más...")
            return False
        except Exception as e:
            print(f"❌ Error al cargar el modelo 3D: {e}")
            
            # Si es un archivo .blend, dar instrucciones específicas
            if file_ext == '.blend':
                print("\n💡 Para usar archivos .blend:")
                print("  Opción 1: Exporta desde Blender como OBJ, STL o GLB")
                print("  Opción 2: Ejecuta desde Blender:")
                print("    blender --background --python visualizer_3d.py -- csv.csv eye.blend")
            
            return False
    
    def load_data(self):
        """Carga los datos del archivo CSV."""
        if not os.path.exists(self.csv_file):
            print(f"Error: El archivo '{self.csv_file}' no existe.")
            return False
        
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"Datos cargados: {len(self.data)} frames")
            
            # Filtrar frames con datos válidos
            self.data = self.data.dropna(subset=['sphere_center_x', 'gaze_x'])
            print(f"Frames válidos: {len(self.data)}")
            
            if len(self.data) == 0:
                print("Error: No hay datos válidos en el CSV.")
                return False
                
            return True
        except Exception as e:
            print(f"Error al cargar el archivo CSV: {e}")
            return False
    
    def create_sphere(self, center, radius, color='lightblue', alpha=0.3):
        """Crea una esfera en 3D."""
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        return self.ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True)
    
    def create_eye_from_mesh(self, center, rotation_matrix=None):
        """Crea el ojo usando el mesh de Blender."""
        if self.eye_mesh_vertices is None:
            return self.create_sphere(center, 1.0 / 1.05, color='lightblue', alpha=0.4)
        
        # Escalar el mesh al tamaño correcto
        eye_radius = 1.0 / 1.05
        vertices = self.eye_mesh_vertices.copy()
        
        # Normalizar y escalar
        center_mesh = vertices.mean(axis=0)
        vertices -= center_mesh
        max_dist = np.max(np.linalg.norm(vertices, axis=1))
        vertices = vertices / max_dist * eye_radius
        
        # Aplicar rotación si se proporciona
        if rotation_matrix is not None:
            vertices = vertices @ rotation_matrix.T
        
        # Trasladar al centro del ojo
        vertices += center
        
        # Crear colección de polígonos
        faces_vertices = []
        for face in self.eye_mesh_faces:
            face_verts = vertices[face]
            faces_vertices.append(face_verts)
        
        # Crear colección 3D
        collection = Poly3DCollection(faces_vertices, alpha=0.6, 
                                     facecolor='lightblue', edgecolor='darkblue', linewidth=0.5)
        self.ax.add_collection3d(collection)
        
        return collection
    
    def create_pupil_ellipse(self, sphere_center, gaze_direction, ellipse_width, ellipse_height, angle):
        """Crea una elipse en la superficie de la esfera representando la pupila."""
        eye_radius = 1.0 / 1.05
        
        # Punto donde la mirada intersecta la esfera (centro de la pupila)
        pupil_center = sphere_center + gaze_direction * eye_radius
        
        # Escalar el tamaño de la elipse
        scale_factor = 0.003
        width_3d = ellipse_width * scale_factor
        height_3d = ellipse_height * scale_factor
        
        # Crear la elipse en el plano local
        theta = np.linspace(0, 2 * np.pi, 50)
        ellipse_x = width_3d * np.cos(theta)
        ellipse_y = height_3d * np.sin(theta)
        
        # Rotar la elipse según el ángulo
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        ellipse_x_rot = ellipse_x * cos_a - ellipse_y * sin_a
        ellipse_y_rot = ellipse_x * sin_a + ellipse_y * cos_a
        
        # Crear sistema de coordenadas local
        z_local = gaze_direction / np.linalg.norm(gaze_direction)
        
        # **CAMBIO CLAVE PARA Y-Z INTERCAMBIADO**
        # El vector 'Up' debe apuntar en la dirección vertical, que ahora es el eje Y (índice 1).
        if abs(z_local[1]) < 0.9: 
            # Si no miramos casi hacia arriba o abajo (eje Y), usamos el vector 'Up' local [0, 1, 0]
            x_local = np.cross([0, 1, 0], z_local)
        else:
            # Si miramos casi vertical, usamos el eje X local [1, 0, 0] para evitar colinealidad
            x_local = np.cross([1, 0, 0], z_local)
            
        x_local = x_local / np.linalg.norm(x_local)
        
        y_local = np.cross(z_local, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        
        # Transformar puntos de la elipse
        ellipse_points = []
        for i in range(len(theta)):
            point_local = (ellipse_x_rot[i] * x_local + 
                           ellipse_y_rot[i] * y_local)
            point_world = pupil_center + point_local
            ellipse_points.append(point_world)
        
        ellipse_array = np.array(ellipse_points)
        
        # Dibujar la elipse
        line = self.ax.plot(ellipse_array[:, 0], ellipse_array[:, 1], 
                            ellipse_array[:, 2], 'k-', linewidth=2, label='Pupil')[0]
        
        # Rellenar la elipse
        verts = [ellipse_array]
        poly = Poly3DCollection(verts, alpha=0.9, facecolor='black', edgecolor='black')
        self.ax.add_collection3d(poly)
        
        return line, poly
    
    def setup_plot(self):
        """Configura la figura y los ejes 3D - Ejes Y y Z intercambiados visualmente."""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Los límites se intercambian para que Y sea vertical y Z sea profundidad
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])  # Y visual (Altura)
        self.ax.set_zlim([-1, 3])  # Z visual (Profundidad)
        
        # **CAMBIO DE ETIQUETAS**
        self.ax.set_xlabel('X (Horizontal →)', fontsize=10)
        self.ax.set_ylabel('Z (Profundidad →)', fontsize=10) # Y ahora es Z
        self.ax.set_zlabel('Y (Altura →)', fontsize=10) # Z ahora es Y en la visualización
        self.ax.set_title('Visualización 3D Eye Tracking', fontsize=12)
        
        # Ajustar ángulo de vista
        self.ax.view_init(elev=20, azim=45)
        
        # Texto informativo
        self.frame_text = self.fig.text(0.02, 0.98, '', fontsize=9, 
                                        verticalalignment='top', family='monospace',
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Grid
        self.ax.grid(True, alpha=0.3)
        
        # Ejes de referencia
        axis_length = 1.5
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.6)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.6)  # Y verde (Altura)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.6)  # Z azul (Profundidad)
        
        # Etiquetas de ejes
        self.ax.text(axis_length, 0, 0, 'X', color='red', fontsize=12, weight='bold')
        self.ax.text(0, axis_length, 0, 'Z', color='green', fontsize=12, weight='bold')
        self.ax.text(0, 0, axis_length, 'Y', color='blue', fontsize=12, weight='bold')
    
    def init_animation(self):
        """Inicializa la animación."""
        self.eye_sphere = None
        self.pupil_ellipse = None
        
        self.gaze_line, = self.ax.plot([], [], [], 'r-', linewidth=3, label='Dirección de Mirada')
        self.gaze_point, = self.ax.plot([], [], [], 'ro', markersize=10, label='Punto de Mirada')
        self.trail_line, = self.ax.plot([], [], [], 'orange', linewidth=2, alpha=0.6, label='Trayectoria')
        
        self.ax.legend(loc='upper left', fontsize=9)
        
        return self.gaze_line, self.gaze_point, self.trail_line
    
    def swap_yz_coords(self, vector):
        """Intercambia las coordenadas Y y Z SOLO PARA VISUALIZACIÓN (Y_csv -> Z_plot, Z_csv -> Y_plot)."""
        # (X, Y_csv, Z_csv) -> (X, Z_csv, Y_csv) para que Z_csv sea Altura (Y_plot) y Y_csv sea Profundidad (Z_plot)
        return np.array([vector[0], vector[2], vector[1]])
    
    def update_animation(self, frame_idx):
        """Actualiza la animación para cada frame."""
        if frame_idx >= len(self.data):
            frame_idx = len(self.data) - 1
        
        row = self.data.iloc[frame_idx]
        
        # Obtener datos y **INVERTIR Y y Z para visualización**
        sphere_center_original = np.array([
            row['sphere_center_x'],
            row['sphere_center_y'],
            row['sphere_center_z']
        ])
        sphere_center = self.swap_yz_coords(sphere_center_original)
        
        gaze_direction_original = np.array([
            row['gaze_x'],
            row['gaze_y'],
            row['gaze_z']
        ])
        gaze_direction = self.swap_yz_coords(gaze_direction_original)
        
        # Datos de la elipse
        ellipse_width = row['ellipse_width'] if not pd.isna(row['ellipse_width']) else 50
        ellipse_height = row['ellipse_height'] if not pd.isna(row['ellipse_height']) else 50
        ellipse_angle = row['ellipse_angle'] if not pd.isna(row['ellipse_angle']) else 0
        
        # Limpiar elementos anteriores
        if self.eye_sphere is not None:
            if hasattr(self.eye_sphere, 'remove'):
                try:
                    self.eye_sphere.remove()
                except Exception:
                    pass
        
        if self.pupil_ellipse is not None:
            line, poly = self.pupil_ellipse
            if line in self.ax.lines:
                line.remove()
            if hasattr(poly, 'remove'):
                try:
                    poly.remove()
                except Exception:
                    pass
        
        # Crear ojo (mesh de Blender o esfera)
        eye_radius = 1.0 / 1.05
        if self.eye_mesh_vertices is not None:
            self.eye_sphere = self.create_eye_from_mesh(sphere_center)
        else:
            self.eye_sphere = self.create_sphere(sphere_center, eye_radius, color='lightblue', alpha=0.4)
        
        # Crear elipse de la pupila
        self.pupil_ellipse = self.create_pupil_ellipse(sphere_center, gaze_direction, 
                                                       ellipse_width, ellipse_height, ellipse_angle)
        
        # Vector de mirada
        gaze_length = 2.0
        gaze_end = sphere_center + gaze_direction * gaze_length
        
        self.gaze_line.set_data([sphere_center[0], gaze_end[0]], 
                                 [sphere_center[1], gaze_end[1]])
        self.gaze_line.set_3d_properties([sphere_center[2], gaze_end[2]])
        
        self.gaze_point.set_data([gaze_end[0]], [gaze_end[1]])
        self.gaze_point.set_3d_properties([gaze_end[2]])
        
        # Trail
        self.trail_points.append(gaze_end.copy())
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
        
        if len(self.trail_points) > 1:
            trail_array = np.array(self.trail_points)
            self.trail_line.set_data(trail_array[:, 0], trail_array[:, 1])
            self.trail_line.set_3d_properties(trail_array[:, 2])
        
        # Texto informativo (mostrar valores ORIGINALES del CSV)
        info_text = f"Frame: {int(row['frame_number'])} / {len(self.data)-1}\n"
        info_text += f"Tiempo: {row['timestamp']:.3f}s\n\n"
        info_text += "Centro Globo Ocular (Original):\n"
        info_text += f"   X (Horizontal): {sphere_center_original[0]:7.3f}\n"
        info_text += f"   Y (Profundidad): {sphere_center_original[1]:7.3f}\n" # Y original
        info_text += f"   Z (Altura): {sphere_center_original[2]:7.3f}\n\n"    # Z original
        info_text += "Dirección Mirada (Original):\n"
        info_text += f"   X (Horizontal): {gaze_direction_original[0]:7.3f}\n"
        info_text += f"   Y (Profundidad): {gaze_direction_original[1]:7.3f}\n"
        info_text += f"   Z (Altura): {gaze_direction_original[2]:7.3f}\n\n"
        info_text += "Elipse Pupila:\n"
        info_text += f"   Ancho:  {ellipse_width:.1f}px\n"
        info_text += f"   Alto:   {ellipse_height:.1f}px\n"
        info_text += f"   Ángulo: {ellipse_angle:.1f}°"
        
        self.frame_text.set_text(info_text)
        
        return self.gaze_line, self.gaze_point, self.trail_line
    
    def animate(self, interval=50, save_as=None):
        """Inicia la animación."""
        self.setup_plot()
        self.init_animation()
        
        anim = FuncAnimation(
            self.fig,
            self.update_animation,
            frames=len(self.data),
            interval=interval,
            blit=False,
            repeat=True
        )
        
        if save_as:
            print(f"Guardando animación como {save_as}...")
            anim.save(save_as, writer='pillow', fps=20)
            print("Animación guardada.")
        
        plt.show()
    
    def plot_static_trajectory(self):
        """Crea un gráfico estático de la trayectoria completa."""
        self.setup_plot()
        
        gaze_points = []
        sphere_centers = []
        
        for idx, row in self.data.iterrows():
            sphere_center_orig = np.array([
                row['sphere_center_x'],
                row['sphere_center_y'],
                row['sphere_center_z']
            ])
            sphere_center = self.swap_yz_coords(sphere_center_orig)
            
            gaze_direction_orig = np.array([
                row['gaze_x'],
                row['gaze_y'],
                row['gaze_z']
            ])
            gaze_direction = self.swap_yz_coords(gaze_direction_orig)
            
            gaze_length = 2.0
            gaze_end = sphere_center + gaze_direction * gaze_length
            
            gaze_points.append(gaze_end)
            sphere_centers.append(sphere_center)
        
        gaze_array = np.array(gaze_points)
        sphere_array = np.array(sphere_centers)
        
        # Ojo promedio
        avg_sphere_center = np.mean(sphere_array, axis=0)
        eye_radius = 1.0 / 1.05
        
        if self.eye_mesh_vertices is not None:
            self.create_eye_from_mesh(avg_sphere_center)
        else:
            self.create_sphere(avg_sphere_center, eye_radius, color='lightblue', alpha=0.3)
        
        # Trayectoria con gradiente
        n_points = len(gaze_array)
        for i in range(n_points - 1):
            color_intensity = i / n_points
            self.ax.plot(gaze_array[i:i+2, 0], gaze_array[i:i+2, 1], gaze_array[i:i+2, 2], 
                         color=(1.0, color_intensity, 0), linewidth=2, alpha=0.7)
        
        # Puntos inicial y final
        self.ax.plot([gaze_array[0, 0]], [gaze_array[0, 1]], [gaze_array[0, 2]], 
                     'go', markersize=12, label='Inicio', zorder=5)
        self.ax.plot([gaze_array[-1, 0]], [gaze_array[-1, 1]], [gaze_array[-1, 2]], 
                     'ro', markersize=12, label='Final', zorder=5)
        
        self.ax.legend(fontsize=10)
        self.ax.set_title('Trayectoria Completa de Eye Tracking (Y=Altura, Z=Profundidad)', fontsize=12)
        
        stats_text = f"Total frames: {len(self.data)}\n"
        stats_text += f"Duración: {self.data['timestamp'].iloc[-1]:.2f}s\n"
        stats_text += "Centro promedio (valores originales):\n"
        
        # Convertir de vuelta al formato original para el texto informativo
        avg_original = self.swap_yz_coords(avg_sphere_center)
        stats_text += f"   X: {avg_original[0]:.3f}\n"
        stats_text += f"   Y (Profundidad): {avg_original[2]:.3f}\n" # El Y original es el Z visual
        stats_text += f"   Z (Altura): {avg_original[1]:.3f}" # El Z original es el Y visual
        
        if self.eye_mesh_vertices is not None:
            stats_text += "\n\n✅ Usando modelo 3D personalizado"
        
        self.fig.text(0.02, 0.5, stats_text, fontsize=9, 
                      verticalalignment='center', family='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.show()

def main():
    """Función principal."""
    print("=" * 60)
    print("VISUALIZADOR 3D DE EYE TRACKING")
    print("Visualización: Y (Altura), Z (Profundidad)")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        blend_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        csv_file = input("\nIngresa la ruta del archivo CSV: ").strip().strip('"')
        blend_file = input("Ingresa la ruta del modelo 3D (.obj, .stl, .ply, .glb) (opcional): ").strip().strip('"')
        if not blend_file:
            blend_file = None
    
    visualizer = EyeTrackingVisualizer(csv_file, blend_file)
    
    # Intentar cargar modelo 3D
    if blend_file:
        print("\n" + "=" * 60)
        if visualizer.load_3d_model():
            print("✅ Modelo 3D cargado correctamente")
        else:
            print("⚠️  Usando esfera por defecto")
        print("=" * 60)
    
    if not visualizer.load_data():
        return
    
    print("\n" + "=" * 60)
    print("OPCIONES DE VISUALIZACIÓN")
    print("=" * 60)
    print("1. Animación en tiempo real (recomendado)")
    print("2. Trayectoria estática completa")
    print("3. Guardar animación como GIF")
    print("=" * 60)
    
    choice = input("\nSelecciona una opción (1-3): ").strip()
    
    if choice == '1':
        print("\n🎬 Iniciando animación...")
        print("💡 Tip: Puedes rotar la vista con el mouse")
        print("❌ Cierra la ventana para terminar.\n")
        visualizer.animate(interval=50)
    elif choice == '2':
        print("\n📊 Mostrando trayectoria completa...")
        visualizer.plot_static_trajectory()
    elif choice == '3':
        output_file = input("\nNombre del archivo GIF (default: eye_tracking.gif): ").strip()
        if not output_file:
            output_file = "eye_tracking.gif"
        if not output_file.endswith('.gif'):
            output_file += '.gif'
        visualizer.animate(interval=50, save_as=output_file)
    else:
        print("❌ Opción no válida.")

if __name__ == "__main__":
    main()