import tkinter as tk
import threading
import pyautogui

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()

class CalibrationOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'white')
        self.root.configure(bg='white')
        self.root.overrideredirect(True)
        
        # Ocultar la barra de título y hacer transparente
        self.root.overrideredirect(True)
        
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Solo el punto central
        self.calibration_point = (MONITOR_WIDTH // 2, MONITOR_HEIGHT // 2)
        
        self.point_visible = False
        self.countdown = 3  # Segundos para mirar el punto
        self.calibration_complete = False
        
    def show_calibration_point(self):
        """Muestra el punto de calibración central con cuenta regresiva"""
        x, y = self.calibration_point
        
        # Limpiar canvas
        self.canvas.delete("all")
        
        # Dibujar cruz grande y visible
        cross_size = 60  # Más grande para mejor visibilidad
        self.canvas.create_line(x - cross_size, y, x + cross_size, y, 
                               fill='red', width=6, tags="cross")
        self.canvas.create_line(x, y - cross_size, x, y + cross_size, 
                               fill='red', width=6, tags="cross")
        
        # Círculo exterior más grande
        circle_radius = 80
        self.canvas.create_oval(x - circle_radius, y - circle_radius, 
                               x + circle_radius, y + circle_radius, 
                               outline='red', width=3, tags="circle")
        
        # Texto de instrucción más prominente
        self.canvas.create_text(x, y + 120, text="MIRA EL PUNTO ROJO", 
                               fill='red', font=('Arial', 18, 'bold'), tags="instruction")
        
        self.point_visible = True
        self.start_countdown()
    
    def start_countdown(self):
        """Inicia cuenta regresiva para el punto"""
        self.countdown = 5
        self.update_countdown()
    
    def update_countdown(self):
        """Actualiza la cuenta regresiva"""
        x, y = self.calibration_point
        
        if self.countdown > 0 and self.point_visible:
            # Actualizar texto de cuenta regresiva
            self.canvas.delete("countdown")
            self.canvas.create_text(x, y + 160, text=f"{self.countdown}", 
                                   fill='red', font=('Arial', 24, 'bold'), tags="countdown")
            
            self.countdown -= 1
            self.root.after(1000, self.update_countdown)
        elif self.countdown == 0 and self.point_visible:
            # Tiempo completado, cerrar overlay
            self.complete_calibration()
    
    def complete_calibration(self):
        """Completa la calibración y cierra el overlay"""
        self.calibration_complete = True
        self.hide_all()
        print("[Calibración] Punto central capturado - Overlay cerrado")
    
    def hide_all(self):
        """Oculta todos los elementos y cierra la ventana"""
        self.canvas.delete("all")
        self.point_visible = False
        # Cerrar la ventana después de un breve delay
        self.root.after(500, self.root.destroy)
    
    def start_calibration(self):
        """Inicia la calibración del punto central"""
        self.show_calibration_point()
        self.root.mainloop()

# Variable global para la overlay
calibration_overlay = None

def start_center_calibration():
    """Inicia la calibración visual del centro en un hilo separado"""
    global calibration_overlay
    
    def show_overlay():
        global calibration_overlay
        calibration_overlay = CalibrationOverlay()
        calibration_overlay.start_calibration()
    
    overlay_thread = threading.Thread(target=show_overlay)
    overlay_thread.daemon = True
    overlay_thread.start()
    
    print("[Calibración] Overlay iniciado - Mira el punto rojo central por 5 segundos")
    return overlay_thread

def is_calibration_overlay_active():
    """Verifica si el overlay de calibración está activo"""
    global calibration_overlay
    return calibration_overlay is not None and calibration_overlay.point_visible

def get_calibration_point():
    """Obtiene las coordenadas del punto de calibración"""
    global calibration_overlay
    if calibration_overlay:
        return calibration_overlay.calibration_point
    return (MONITOR_WIDTH // 2, MONITOR_HEIGHT // 2)


def show_multipoint():
    """muestra 5 puntos para extraer datos"""
    global calibration_overlay
    
    def show_overlay():
        global calibration_overlay
        calibration_overlay = CalibrationOverlay()
        calibration_overlay.start_calibration()
    
    overlay_thread = threading.Thread(target=show_overlay)
    overlay_thread.daemon = True
    overlay_thread.start()
    
    print("[Calibración] Overlay iniciado - Mira el punto rojo central por 5 segundos")
    return overlay_thread