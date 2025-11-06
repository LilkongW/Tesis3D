import cv2
from ultralytics import YOLO
import os
import numpy as np # Necesitamos numpy para el manejo de arrays

# --- CONFIGURACIÓN ---

# ⚠️ USA "r" ANTES DEL STRING para que Python lea bien las rutas de Windows
MODEL_PATH = r"C:\Users\Victor\Documents\Tesis3D\models\best.pt"

# ⚠️ CAMBIA ESTO por la ruta a tu video de entrada
VIDEO_IN_PATH = r"C:\Users\Victor\Documents\Tesis3D\Videos\Experimento_1\Venegas\Venegas_intento_1.mp4" 

# ⚠️ CAMBIA ESTO por cómo quieres que se llame el video de salida
VIDEO_OUT_PATH = r"C:\Users\Victor\Documents\Tesis3D\Videos\resultado_con_elipse.mp4"

# ---------------------

# 1. Cargar el modelo
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit()

# 2. Abrir el video de entrada
cap = cv2.VideoCapture(VIDEO_IN_PATH)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video de entrada en: {VIDEO_IN_PATH}")
    exit()

# 3. Obtener propiedades del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 4. Crear el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (frame_width, frame_height))

print("Procesando video con ajuste de elipse... Presiona 'q' en la ventana para salir.")

# 5. Bucle principal para procesar cada frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 6. Ejecutar la inferencia (Tracking)
    results = model.track(frame, persist=True)

    # --- INICIO DE LA MODIFICACIÓN ---
    
    # 7. DIBUJAR RESULTADOS MANUALMENTE (YA NO USAMOS results[0].plot())
    
    # Creamos una copia del frame original para dibujar sobre ella
    annotated_frame = frame.copy()

    # Verificamos si hay alguna detección en esta frame
    if results[0].boxes is not None:
        
        # Obtenemos todas las cajas (bboxes) detectadas
        boxes = results[0].boxes
        
        # Iteramos sobre cada una de las cajas (ROIs)
        for box in boxes:
            
            # --- 7a. Obtener el ROI de YOLO ---
            
            # Extraer coordenadas [x1, y1, x2, y2] y convertirlas a enteros
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # (Opcional) Dibujar el Bounding Box original de YOLO en azul
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # Asegurarnos de que las coordenadas no se salgan de los límites del frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            
            # Omitir si la caja no tiene área (evita errores)
            if x1 >= x2 or y1 >= y2:
                continue
                
            # Recortar el ROI (Region of Interest) del frame original
            roi = frame[y1:y2, x1:x2]

            # --- 7b. Binarización del ROI ---
            
            # Convertir el ROI a escala de grises
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Aplicar binarización. 
            # Usamos THRESH_BINARY_INV asumiendo que buscas un objeto oscuro (pupila)
            # en un fondo claro. Si es al revés, usa THRESH_BINARY.
            # Otsu (THRESH_OTSU) encuentra el umbral óptimo automáticamente.
            _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # --- 7c. Encontrar Contornos y Ajustar Elipse ---
            
            # Encontrar contornos en la imagen binarizada
            # RETR_EXTERNAL solo toma los contornos exteriores
            contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Si se encontraron contornos...
            if contours:
                # Encontrar el contorno más grande (por área)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Se necesitan al menos 5 puntos para ajustar una elipse
                if len(largest_contour) >= 5:
                    try:
                        # Ajustar la elipse al contorno más grande
                        ellipse = cv2.fitEllipse(largest_contour)
                        
                        # --- 7d. Dibujar la Elipse (en el frame completo) ---
                        
                        # 'ellipse' tiene coordenadas relativas al ROI.
                        # Necesitamos "sumar" las coordenadas (x1, y1) del ROI
                        # para dibujarla en el frame completo.
                        
                        (center_x_roi, center_y_roi), (axis_len1, axis_len2), angle = ellipse
                        
                        # Calcular el centro absoluto de la elipse
                        center_x_abs = center_x_roi + x1
                        center_y_abs = center_y_roi + y1
                        
                        # Crear el objeto de elipse con coordenadas absolutas
                        abs_ellipse = ((center_x_abs, center_y_abs), (axis_len1, axis_len2), angle)
                        
                        # Dibujar la elipse en el frame anotado (en verde)
                        cv2.ellipse(annotated_frame, abs_ellipse, (0, 255, 0), 2)
                        
                    except cv2.error:
                        # fitEllipse puede fallar en formas muy raras (ej. una línea recta)
                        pass # Simplemente no dibujamos la elipse si falla

    # --- FIN DE LA MODIFICACIÓN ---

    # 8. Guardar el frame anotado (ahora con nuestra elipse y/o caja)
    out.write(annotated_frame)

    # (Opcional) Mostrar el video en una ventana
    cv2.imshow("Deteccion en Video", annotated_frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Liberar recursos
print(f"Proceso completado. Video guardado en: {VIDEO_OUT_PATH}")
cap.release()
out.release()
cv2.destroyAllWindows()