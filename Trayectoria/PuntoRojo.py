import tkinter as tk
import threading
import pyautogui

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()

class MovingPointOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'white')
        self.root.configure(bg='white')
        self.root.overrideredirect(True)
        
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Punto inicial en el centro
        self.current_x = MONITOR_WIDTH // 2
        self.current_y = MONITOR_HEIGHT // 2
        self.point_visible = False
        self.point_id = None
        
    def show_point(self, x=None, y=None):
        """Muestra el punto rojo en las coordenadas especificadas"""
        # Si no se especifican coordenadas, usar las actuales
        if x is not None:
            self.current_x = x
        if y is not None:
            self.current_y = y
            
        # Limpiar canvas
        self.canvas.delete("all")
        
        # Dibujar punto rojo grande
        point_radius = 15
        self.point_id = self.canvas.create_oval(
            self.current_x - point_radius, self.current_y - point_radius,
            self.current_x + point_radius, self.current_y + point_radius,
            fill='red', outline='red', width=3, tags="point"
        )
        
        # Opcional: añadir cruz para mejor visibilidad
        cross_size = 25
        self.canvas.create_line(
            self.current_x - cross_size, self.current_y,
            self.current_x + cross_size, self.current_y,
            fill='red', width=3, tags="cross"
        )
        self.canvas.create_line(
            self.current_x, self.current_y - cross_size,
            self.current_x, self.current_y + cross_size,
            fill='red', width=3, tags="cross"
        )
        
        # Texto con coordenadas (opcional)
        coord_text = f"({self.current_x}, {self.current_y})"
        self.canvas.create_text(
            self.current_x, self.current_y + 40,
            text=coord_text, fill='red', font=('Arial', 12, 'bold'), tags="coords"
        )
        
        self.point_visible = True
        
    def move_point(self, x, y):
        """Mueve el punto a nuevas coordenadas"""
        self.current_x = max(0, min(x, MONITOR_WIDTH - 1))
        self.current_y = max(0, min(y, MONITOR_HEIGHT - 1))
        
        if self.point_visible:
            self.show_point(self.current_x, self.current_y)
        
    def hide_point(self):
        """Oculta el punto"""
        self.canvas.delete("all")
        self.point_visible = False
        
    def destroy(self):
        """Cierra completamente el overlay"""
        self.hide_point()
        self.root.destroy()
        
    def start(self):
        """Inicia el overlay"""
        self.show_point()
        self.root.mainloop()

# Variable global para el overlay móvil
moving_point_overlay = None

def create_moving_point():
    """Crea y muestra el punto móvil en un hilo separado"""
    global moving_point_overlay
    
    def show_overlay():
        global moving_point_overlay
        moving_point_overlay = MovingPointOverlay()
        moving_point_overlay.start()
    
    overlay_thread = threading.Thread(target=show_overlay)
    overlay_thread.daemon = True
    overlay_thread.start()
    
    print("[Moving Point] Overlay móvil iniciado - Punto rojo en el centro")
    return overlay_thread

def move_point_to(x, y):
    """Mueve el punto a coordenadas específicas"""
    global moving_point_overlay
    if moving_point_overlay and moving_point_overlay.point_visible:
        moving_point_overlay.move_point(x, y)
        print(f"[Moving Point] Punto movido a ({x}, {y})")
        return True
    else:
        print("[Moving Point] Error: Overlay no está activo")
        return False

def hide_moving_point():
    """Oculta el punto móvil"""
    global moving_point_overlay
    if moving_point_overlay:
        moving_point_overlay.hide_point()
        print("[Moving Point] Punto ocultado")

def destroy_moving_point():
    """Destruye completamente el overlay móvil"""
    global moving_point_overlay
    if moving_point_overlay:
        moving_point_overlay.destroy()
        moving_point_overlay = None
        print("[Moving Point] Overlay destruido")

def is_moving_point_visible():
    """Verifica si el punto móvil está visible"""
    global moving_point_overlay
    return moving_point_overlay is not None and moving_point_overlay.point_visible

def get_current_point_position():
    """Obtiene la posición actual del punto"""
    global moving_point_overlay
    if moving_point_overlay:
        return (moving_point_overlay.current_x, moving_point_overlay.current_y)
    return (MONITOR_WIDTH // 2, MONITOR_HEIGHT // 2)