"""
Script completo para mejorar videos de baja calidad usando ESRGAN
Aumenta la resolución y mejora los detalles del video, con opción de
mantener la resolución original (mejorando solo la calidad).

Instalación requerida:
pip install opencv-python numpy torch torchvision basicsr realesrgan pillow tqdm
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import os
from tqdm import tqdm
import argparse

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_AVAILABLE = True
except ImportError:
    ESRGAN_AVAILABLE = False
    print("Error: No se encontró RealESRGAN")
    print("Instala con: pip install basicsr realesrgan")


class VideoUpscaler:
    """
    Mejora videos usando ESRGAN para super-resolución
    """
    
    def __init__(self, model_name='RealESRGAN_x4plus', device='auto', tile_size=512):
        """
        Args:
            model_name: Modelo a usar ('RealESRGAN_x4plus', 'RealESRGAN_x2plus', 'RealESRNet_x4plus')
            device: 'cuda', 'cpu', o 'auto'
            tile_size: Tamaño de tile (más grande = más rápido pero más VRAM)
        """
        if not ESRGAN_AVAILABLE:
            raise ImportError("RealESRGAN no está instalado")
        
        # Configurar dispositivo
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"🖥️  Usando dispositivo: {self.device}")
        
        # Configurar modelo
        self.model_name = model_name
        self.upsampler = None
        self.tile_size = tile_size
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo ESRGAN"""
        print(f"📦 Cargando modelo {self.model_name}...")
        
        # Configuración según el modelo
        model_configs = {
            'RealESRGAN_x4plus': {
                'scale': 4,
                'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                'num_block': 23,
                'num_feat': 64
            },
            'RealESRGAN_x2plus': {
                'scale': 2,
                'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                'num_block': 23,
                'num_feat': 64
            },
            'RealESRNet_x4plus': {
                'scale': 4,
                'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
                'num_block': 23,
                'num_feat': 64
            },
            'RealESRGAN_x4plus_anime_6B': {
                'scale': 4,
                'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
                'num_block': 6,
                'num_feat': 64
            }
        }
        
        if self.model_name not in model_configs:
            raise ValueError(f"Modelo no soportado: {self.model_name}")
        
        config = model_configs[self.model_name]
        
        # Crear modelo
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=config['num_feat'],
            num_block=config['num_block'],
            num_grow_ch=32,
            scale=config['scale']
        )
        
        # Descargar modelo si es necesario
        model_path = Path('models') / f"{self.model_name}.pth"
        model_path.parent.mkdir(exist_ok=True)
        
        if not model_path.exists():
            print(f"📥 Descargando modelo desde {config['model_path']}...")
            import urllib.request
            urllib.request.urlretrieve(config['model_path'], str(model_path))
            print("✓ Modelo descargado")
        
        # Inicializar upsampler
        self.upsampler = RealESRGANer(
            scale=config['scale'],
            model_path=str(model_path),
            model=model,
            tile=self.tile_size,
            tile_pad=10,
            pre_pad=0,
            half=True if self.device == 'cuda' else False,
            device=self.device
        )
        
        print("✓ Modelo cargado exitosamente")
    
    def enhance_frame(self, frame):
        """
        Mejora un frame individual
        
        Args:
            frame: Imagen BGR (numpy array)
        
        Returns:
            Frame mejorado (numpy array)
        """
        try:
            # ESRGAN espera RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar
            output, _ = self.upsampler.enhance(frame_rgb, outscale=None)
            
            # Convertir de vuelta a BGR
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            return output_bgr
        
        except Exception as e:
            print(f"⚠️ Error al procesar frame: {e}")
            return frame
    
    def process_video(self, input_path, output_path, start_frame=0, end_frame=None, 
                      target_size=None, show_preview=True, save_frames=False):
        """
        Procesa un video completo
        
        Args:
            input_path: Ruta del video de entrada
            output_path: Ruta del video de salida
            start_frame: Frame inicial (default: 0)
            end_frame: Frame final (default: None = hasta el final)
            target_size: (width, height) para redimensionar salida (default: None).
                         Puede ser una tupla (W, H) o la cadena 'original'.
            show_preview: Mostrar preview durante procesamiento
            save_frames: Guardar frames individuales
        """
        # Abrir video
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {input_path}")
        
        # Obtener propiedades del video original (Variables solicitadas)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # === VARIABLES ORIGINALES EXTRAÍDAS ===
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # =====================================

        
        # === LÓGICA PARA MANTENER LA RESOLUCIÓN ORIGINAL (O forzar una nueva) ===
        target_size_tuple = None
        if isinstance(target_size, str) and target_size.lower() == 'original':
             # 1. Si se pide 'original', la nueva dimensión es la original.
             target_size_tuple = (width, height)
             print(f"✅ Opción: Forzar la resolución de salida a la original ({width}x{height})")
        elif isinstance(target_size, tuple) and len(target_size) == 2:
            # 2. Si se pasa una tupla (W, H)
            target_size_tuple = target_size
            
        # ========================================================================
        
        if end_frame is None:
            end_frame = total_frames
        else:
            end_frame = min(end_frame, total_frames)
        
        frames_to_process = end_frame - start_frame
        
        print(f"\n{'='*60}")
        print("📹 Información del video:")
        print(f"   Resolución original: {width}x{height}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Total frames: {total_frames}")
        print(f"   Frames a procesar: {frames_to_process} (del {start_frame} al {end_frame})")
        print(f"{'='*60}\n")
        
        # Saltar al frame inicial
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Procesar primer frame para obtener tamaño de salida
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("No se pudo leer el primer frame")
        
        enhanced_first = self.enhance_frame(first_frame)
        out_height, out_width = enhanced_first.shape[:2]
        
        # Aplicar target_size si se especificó (incluyendo 'original')
        if target_size_tuple is not None:
            out_width, out_height = target_size_tuple
            enhanced_first = cv2.resize(enhanced_first, (out_width, out_height))
        
        print(f"✓ Resolución de salida: {out_width}x{out_height}")
        
        # Crear directorio para frames si es necesario
        if save_frames:
            frames_dir = Path(output_path).parent / 'enhanced_frames'
            frames_dir.mkdir(exist_ok=True)
            print(f"✓ Guardando frames en: {frames_dir}")
        
        # Configurar writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (out_width, out_height)
        )
        
        # Resetear al frame inicial
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Procesar frames
        print("\n🎬 Procesando video...")
        
        with tqdm(total=frames_to_process, desc="Progreso", unit="frames") as pbar:
            for frame_idx in range(frames_to_process):
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Mejorar frame
                enhanced = self.enhance_frame(frame)
                
                # Redimensionar si es necesario
                if target_size_tuple is not None:
                    enhanced = cv2.resize(enhanced, (out_width, out_height))
                
                # Guardar frame
                out.write(enhanced)
                
                # Guardar frame individual
                if save_frames:
                    frame_path = frames_dir / f"frame_{start_frame + frame_idx:06d}.png"
                    cv2.imwrite(str(frame_path), enhanced)
                
                # Preview
                if show_preview and frame_idx % 10 == 0:
                    # Crear comparación lado a lado
                    frame_resized = cv2.resize(frame, (out_width, out_height))
                    comparison = np.hstack([frame_resized, enhanced])
                    
                    # Redimensionar para visualización
                    display_width = 1280
                    scale = display_width / comparison.shape[1]
                    display_size = (display_width, int(comparison.shape[0] * scale))
                    comparison_display = cv2.resize(comparison, display_size)
                    
                    # Añadir texto
                    cv2.putText(comparison_display, "Original", (10, 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(comparison_display, "Mejorado", (display_width//2 + 10, 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    #cv2.imshow("Comparación: Original vs Mejorado", comparison_display)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⚠️ Procesamiento cancelado por el usuario")
                        break
                
                pbar.update(1)
        
        # Limpiar
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Video procesado y guardado en: {output_path}")
        
        # Mostrar estadísticas
        output_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✓ Tamaño del archivo: {output_size:.2f} MB")


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Mejora videos usando ESRGAN')
    
    parser.add_argument('input', type=str, help='Ruta del video de entrada')
    parser.add_argument('output', type=str, help='Ruta del video de salida')
    parser.add_argument('--model', type=str, default='RealESRGAN_x4plus',
                         choices=['RealESRGAN_x4plus', 'RealESRGAN_x2plus', 
                                  'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B'],
                         help='Modelo a usar')
    parser.add_argument('--device', type=str, default='auto',
                         choices=['auto', 'cuda', 'cpu'],
                         help='Dispositivo (auto, cuda, cpu)')
    parser.add_argument('--tile-size', type=int, default=512,
                         help='Tamaño de tile (más grande = más rápido pero más VRAM)')
    parser.add_argument('--start-frame', type=int, default=0,
                         help='Frame inicial')
    parser.add_argument('--end-frame', type=int, default=None,
                         help='Frame final (None = hasta el final)')
    # === ARGUMENTO DE LÍNEA DE COMANDOS CORREGIDO Y UNIFICADO ===
    parser.add_argument('--target-size', type=str, default=None,
                         help="Resolución de salida deseada (ej: '1920x1080'). Usa 'original' para mantener las dimensiones de entrada.")
    parser.add_argument('--no-preview', action='store_true',
                         help='No mostrar preview durante procesamiento')
    parser.add_argument('--save-frames', action='store_true',
                         help='Guardar frames individuales')
    # ===========================================================
    
    args = parser.parse_args()
    
    # Validar entrada
    if not Path(args.input).exists():
        print(f"❌ Error: El archivo {args.input} no existe")
        return
    
    # Configurar target_size a partir del nuevo argumento unificado
    target_size_config = None
    if args.target_size:
        if args.target_size.lower() == 'original':
            target_size_config = 'original'
        else:
            try:
                w, h = map(int, args.target_size.lower().split('x'))
                target_size_config = (w, h)
            except ValueError:
                print("❌ Error: Formato de --target-size inválido. Debe ser 'ANCHOxALTO' o 'original'.")
                return

    
    # Crear upscaler
    try:
        upscaler = VideoUpscaler(
            model_name=args.model,
            device=args.device,
            tile_size=args.tile_size
        )
        
        # Procesar video
        upscaler.process_video(
            input_path=args.input,
            output_path=args.output,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            target_size=target_size_config, # Se pasa la configuración de target_size
            show_preview=not args.no_preview,
            save_frames=args.save_frames
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# Ejemplo de uso programático
def example_usage():
    """Ejemplo de cómo usar la clase directamente"""
    
    # Crear upscaler
    upscaler = VideoUpscaler(
        model_name='RealESRGAN_x2plus', # x2plus es más rápido y suficiente si se re-escala a 1x
        device='auto',  
        tile_size=512   
    )
    
    # Procesar video
    upscaler.process_video(
        input_path=r'C:\\Users\\Victor\\Documents\\Tesis3D\\Videos\\Experimento_1\\Victor\\ROI_videos_640x480\\Victor_intento_1_ROI_640x480.mp4',
        output_path=r'C:\\Users\\Victor\\Documents\\Tesis3D\\Videos\\Experimento_1\\Victor\\ROI_videos_640x480\\Victor_intento_1_ROI_640x480_dif.mp4',
        start_frame=0,
        end_frame=None,
        # === USO PROGRAMÁTICO PARA MANTENER RESOLUCIÓN ORIGINAL ===
        target_size='original', # <--- ¡Mantiene la resolución original (W x H) del video!
        # ========================================================
        show_preview=True, 
        save_frames=False 
    )


if __name__ == "__main__":
    
    # O descomentar para usar el ejemplo directo:
    example_usage()
    # Para usar el modo de línea de comandos, comenta la línea anterior y descomenta la siguiente:
    # main()