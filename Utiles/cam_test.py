import cv2

# Abre la cámara (0 = cámara por defecto)
camara = cv2.VideoCapture(1)

if not camara.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("🎥 Presiona 'q' para salir.")

while True:
    ret, frame = camara.read()
    if not ret:
        print("⚠️ No se pudo leer el frame.")
        break

    # Muestra la imagen en una ventana
    cv2.imshow("Camara", frame)

    # Si presionas la tecla 'q', se cierra
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
camara.release()
cv2.destroyAllWindows()