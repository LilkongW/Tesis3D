import subprocess
import os

def video_to_gif_ffmpeg_alta_calidad(video_path, gif_path, fps=25, scale='1080:-1'):
    """
    Convierte un video a GIF usando FFmpeg con parámetros optimizados
    para una mayor calidad (mayor resolución y FPS, a costa de un mayor tamaño de archivo).
    
    Args:
        video_path (str): Ruta al archivo de video de entrada.
        gif_path (str): Ruta donde se guardará el archivo GIF de salida.
        fps (int): Cuadros por segundo del GIF. Aumentado a 25.
        scale (str): Resolución. '1080:-1' establece el ancho a 1080px (Full HD).
    """
    palette_path = 'temp_palette.png'
    
    # 1. Comando para generar la paleta de colores óptima (palettegen)
    cmd_palette = [
        'ffmpeg', '-i', video_path, 
        # Aumentamos FPS y resolución aquí también
        '-vf', f'fps={fps},scale={scale}:flags=lanczos,palettegen',
        '-y', palette_path
    ]
    
    # 2. Comando para aplicar la paleta y generar el GIF (paletteuse)
    # Agregamos la opción 'dither=bayer' para mejorar la gestión de gradientes y evitar bandas de color.
    cmd_gif = [
        'ffmpeg', '-i', video_path, '-i', palette_path,
        '-lavfi', f'fps={fps},scale={scale}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5',
        '-y', gif_path
    ]

    try:
        if not os.path.exists(video_path):
            print(f"Error: El archivo de video no existe en {video_path}")
            return
            
        print("Generando paleta de colores óptima para alta calidad...")
        subprocess.run(cmd_palette, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"Exportando GIF a: {gif_path} (FPS: {fps}, Escala: {scale})")
        subprocess.run(cmd_gif, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Limpieza
        os.remove(palette_path)
        
        print(f"\n✅ ¡GIF creado exitosamente! Se guardó como: {gif_path}")

    except Exception as e:
        # Mantener manejo de errores
        print(f"Ocurrió un error: {e}")

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # La ruta al archivo de video que quieres usar
    input_video = r"c:\Users\Victor\Downloads\Venegas-Victor.mp4" 
    output_gif = "video_a_gif_alta_calidad.gif"
    
    # Usar la nueva función con los parámetros de alta calidad
    video_to_gif_ffmpeg_alta_calidad(input_video, output_gif, fps=25, scale='1080:-1')