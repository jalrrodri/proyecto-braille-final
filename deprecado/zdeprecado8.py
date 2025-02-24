import matplotlib.pyplot as plt
from entrenamiento import model, test_data

# Visualizar precisión y pérdida durante el entrenamiento
plt.figure(figsize=(10, 5))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(model.history.history['accuracy'], label='Training Accuracy')
plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(model.history.history['loss'], label='Training Loss')
plt.plot(model.history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluar el modelo en el conjunto de datos de prueba
model.evaluate(test_data)

# Evaluar el modelo TFLite
model.evaluate_tflite("modeloGuardado/optimizado/efficientdet_lite0/model.tflite", test_data)
