# Importa TensorFlow para construir y entrenar el modelo
# Requiere pip install tensorflow==2.10.0
import tensorflow as tf

# Importa capas de Keras para la arquitectura del modelo
from keras import layers, optimizers

# Importa Path para trabajar con rutas de archivos
from pathlib import Path
from keras.regularizers import l2

# Especifica la ruta al conjunto de datos de imágenes
data_dir = Path("images")

# Importa matplotlib
import matplotlib.pyplot as plt

# Define las dimensiones de las imágenes y el tamaño del lote
image_size = (28, 28)
batch_size = 32
image_height = 28
image_width = 28
num_classes = 26  # Número de clases a predecir

# Definir la forma de entrada y el número de clases
input_shape = (28, 28, 3)  # Ajusta según el tamaño de imagen de entrada

# Crea un generador de datos para los datos de entrenamiento con aumento
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                                shear_range=10,
                                                                validation_split=0.2)

# Carga imágenes de entrenamiento desde el directorio usando el generador de datos
train_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Etiquetas codificadas en one-hot
    subset='training',  # Utiliza solo el subconjunto de entrenamiento
    seed=456  # Establece una semilla para reproducibilidad
)

# Carga imágenes de validación directamente desde el directorio
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    subset='validation',  # Utiliza solo el subconjunto de validación
    validation_split=0.2,  # Utiliza el 20% de los datos para validación
    seed=456,  # Misma semilla para consistencia
    labels="inferred",  # Infiera etiquetas de la estructura del directorio
    label_mode='categorical',  # Etiquetas codificadas en one-hot
    image_size=image_size,
    batch_size=batch_size
)

# Definir la arquitectura del modelo
# Crea el modelo Mobile SSD dentro de tf.keras.Sequential
# Crea el modelo base MobileNetV2 sin capa superior
image_size_new = (32,32)
# Aumentar la complejidad del modelo base.
base_model = tf.keras.applications.MobileNetV3Small(input_shape=image_size_new + (3,), include_top=False)

# Ajustar el modelo base
base_model.trainable = True
fine_tune_at = 100  # Fine-tune from this layer onwards
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Aumentar unidades en la capa densa.
#MODELO KAGGLE
model = tf.keras.Sequential([
    layers.SeparableConv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.SeparableConv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.SeparableConv2D(256,(2,2),activation='relu'),
    layers.GlobalMaxPooling2D(),
    layers.Dense(256),
    layers.LeakyReLU(),
    layers.Dense(64,kernel_regularizer=l2(2e-4)),
    layers.LeakyReLU(),
    layers.Dense(26,activation='softmax'),
    ])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo en el conjunto de datos de entrenamiento, validando en el conjunto de datos de validación
# model.fit(train_ds, epochs=100, validation_data=validation_ds)

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    # Plotear valores de precisión de entrenamiento y validación
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
    # Plotear valores de pérdida de entrenamiento y validación
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
    plt.tight_layout()
    plt.show()

# Entrenar el modelo en el conjunto de datos de entrenamiento, validando en el conjunto de datos de validación
history = model.fit(train_ds, epochs=100, validation_data=validation_ds)

# Llamar la función para plotear el historial de entrenamiento
plot_training_history(history)

# Cargar el conjunto de datos de prueba
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",  # Usar el subconjunto de validación para la prueba
    seed=456,
    labels="inferred",
    label_mode="categorical",
    image_size=image_size,
    batch_size=batch_size
)

# Evaluar el modelo en el conjunto de datos de prueba
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Pérdida en la prueba: {test_loss}')
print(f'Precisión en la prueba: {test_accuracy}')

# Visualización del modelo post evaluación con datos de prueba
def visualize_predictions(model, test_dataset, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i, (images, labels) in enumerate(test_dataset.take(num_samples)):
        predictions = model.predict(images)
        predicted_class = tf.argmax(predictions, axis=1).numpy()
        true_class = tf.argmax(labels, axis=1).numpy()
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[0])
        plt.title(f'Predicho: {predicted_class[0]}, Real: {true_class[0]}')
        plt.axis('off')
    plt.show()

# Llamar la función para visualizar las predicciones
visualize_predictions(model, test_ds)

# Exportar modelo hdf5
model.save('modeloGuardado/raw/modelo.h5')
