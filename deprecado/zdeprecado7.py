import cv2
import tensorflow as tf
import numpy as np

# Cargar el modelo guardado
model = tf.keras.models.load_model('modeloGuardado/raw/modelo.h5')

# Configuración de la webcam
cap = cv2.VideoCapture(0)

# Verifica si la cámara se abre correctamente
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

# Función para preprocesar la imagen para la predicción
def preprocess_image(image):
    image = cv2.resize(image, (28, 28))  # Redimensiona la imagen
    image = image / 255.0  # Normaliza la imagen
    image = np.expand_dims(image, axis=0)  # Añade una dimensión de lote
    return image

while True:
    # Captura frame por frame
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir el frame (fin de la transmisión?). Salida ...")
        break

    # Preprocesa la imagen
    preprocessed_image = preprocess_image(frame)

    # Realiza la predicción
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])

    # Muestra el resultado en la ventana de la webcam
    cv2.putText(frame, f'Predicción: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Muestra la imagen en una ventana
    cv2.imshow('Webcam', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
