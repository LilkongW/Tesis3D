import cv2
import requests
import time
import os

# Configuración
ESP32_IP = "192.168.1.109"
URL = f"http://{ESP32_IP}"

# Variables de control
AWB = True
recording = False
video_writer = None
frame_count = 0
start_time = None
CROP_PIXELS = 20

# Fix para Wayland
os.environ['QT_QPA_PLATFORM'] = 'xcb'

def set_resolution(url: str, index: int = 1, verbose: bool = False):
    """Cambiar resolución de la cámara"""
    try:
        if verbose:
            resolutions = {
                10: "UXGA (1600x1200)",
                9: "SXGA (1280x1024)",
                8: "XGA (1024x768)",
                7: "SVGA (800x600)",
                6: "VGA (640x480)",
                5: "CIF (400x296)",
                4: "QVGA (320x240)",
                3: "HQVGA (240x176)",
                0: "QQVGA (160x120)"
            }
            print("\n📐 Resoluciones disponibles:")
            for idx, res in resolutions.items():
                print(f"  {idx}: {res}")

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            response = requests.get(f"{url}/control?var=framesize&val={index}", timeout=2)
            if response.status_code == 200:
                print(f"✓ Resolución cambiada a índice {index}")
                return True
        else:
            print("✗ Índice inválido")
            return False
    except Exception as e:
        print(f"✗ Error cambiando resolución: {e}")
        return False

def set_awb(url: str, awb: bool):
    """Toggle Auto White Balance"""
    try:
        awb = not awb
        requests.get(f"{url}/control?var=awb&val={1 if awb else 0}", timeout=2)
        print(f"✓ AWB: {'ON' if awb else 'OFF'}")
        return awb
    except Exception as e:
        print(f"✗ Error en AWB: {e}")
        return awb

def main():
    global AWB, recording, video_writer, frame_count, start_time
    
    print("═══════════════════════════════════════")
    print("  ESP32-CAM WebServer Viewer")
    print("═══════════════════════════════════════")
    print(f"  Conectando a: {URL}")
    print("═══════════════════════════════════════\n")

    time.sleep(1)
    
    # Abrir stream
    cap = cv2.VideoCapture(f"{URL}:81/stream")
    
    if not cap.isOpened():
        print("✗ Error: No se pudo conectar al stream")
        print(f"  Verifica que el ESP32-CAM esté en: {ESP32_IP}")
        return
    
    print("✓ Stream conectado exitosamente\n")
    
    print("┌─ CONTROLES ─────────────────────────┐")
    print("│  R - Cambiar resolución             │")
    print("│  A - Toggle Auto White Balance      │")
    print("│  G - Iniciar/Detener grabación      │")
    print("│  ESC - Salir                        │")
    print("└──────────────────────────────────────┘\n")
    
    # Variables para FPS
    fps = 0
    fps_counter = 0
    fps_timer = time.time()
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()

            frame = frame[:, CROP_PIXELS:] 
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = gray_frame
            if not ret:
                print("⚠ Error leyendo frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            fps_counter += 1
            
            # Calcular FPS cada segundo
            current_time = time.time()
            if current_time - fps_timer >= 1.0:
                fps = fps_counter / (current_time - fps_timer)
                fps_counter = 0
                fps_timer = current_time
            
            # Info overlay
            info_text = f"FPS: {fps:.1f} | Frames: {frame_count}"
            if recording:
                info_text += " | REC"
                cv2.circle(frame, (20, 20), 8, 255, -1) 
            
            # Texto con sombra
            cv2.putText(frame, info_text, (12, 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 3) 
            # Texto principal (blanco = 255)
            # ❗ CORRECCIÓN 4: Usar 255 (intensidad blanca) para el texto
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2) 
            # Mostrar frame
            cv2.imshow('ESP32-CAM Stream', frame)
            
            # Grabar si está activado
            if recording and video_writer is not None:
                video_writer.write(frame)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\n✓ Saliendo...")
                break
                
            elif key == ord('r') or key == ord('R'):
                print("\n📐 Ingresa el índice de resolución (0-10):")
                print("  Recomendados: 7=SVGA(800x600), 6=VGA(640x480), 4=QVGA(320x240)")
                try:
                    idx = int(input("Índice: "))
                    if set_resolution(URL, index=idx, verbose=True):
                        # Reiniciar captura
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(f"{URL}:81/stream")
                        print("✓ Stream reiniciado\n")
                except ValueError:
                    print("✗ Entrada inválida\n")
                
        
                
            elif key == ord('a') or key == ord('A'):
                AWB = set_awb(URL, AWB)
                
            elif key == ord('g') or key == ord('G'):
                if not recording:
                    # Iniciar grabación
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"video_{timestamp}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    h, w = frame.shape[:2]
                    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                    recording = True
                    print(f"\n🔴 Grabación iniciada: {filename}")
                else:
                    # Detener grabación
                    recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print("⏹ Grabación detenida\n")
    
    except KeyboardInterrupt:
        print("\n✓ Interrumpido por usuario (Ctrl+C)")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        # Limpiar
        if recording and video_writer is not None:
            video_writer.release()
            print("✓ Video guardado")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Estadísticas
        total_time = time.time() - start_time
        print("\n═══════════════════════════════════════")
        print("  ESTADÍSTICAS FINALES")
        print("═══════════════════════════════════════")
        print(f"  Frames totales: {frame_count}")
        print(f"  Tiempo total: {total_time:.2f} segundos")
        if total_time > 0:
            print("  FPS promedio: {frame_count/total_time:.2f}")
        print("═══════════════════════════════════════\n")

if __name__ == "__main__":
    main()