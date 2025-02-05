# Importa las bibliotecas necesarias
import tensorflow as tf
from keras import layers
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Resizing, GlobalAveragePooling2D, Reshape, Conv2D
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy
from keras.backend import clear_session
from pathlib import Path
import os
import matplotlib.pyplot as plt

# Define las dimensiones de las imágenes y el tamaño del lote
image_size = (28, 28)
batch_size = 32
image_height = 28
image_width = 28
num_classes = 26  # Número de clases a predecir

# Define la forma de entrada y el número de clases
input_shape = (28, 28, 3)  # Ajusta según el tamaño de imagen de entrada

# Especifica la ruta al conjunto de datos de imágenes
data_dir = Path("images")

# Crea un generador de datos para los datos de entrenamiento con aumento
train_datagen = ImageDataGenerator(rotation_range=20,
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

# Define la arquitectura del modelo
# Crea el modelo Mobile SSD dentro de tf.keras.Sequential
modelo = Sequential([
    # Redimensiona las imágenes de entrada a 32x32
    Resizing(32, 32),

    # Utiliza MobileNetV2 como capas base pre-entrenadas
    MobileNetV2(input_shape=(32, 32, 3), include_top=False),

    # Aplica pooling global para obtener una representación vectorial
    GlobalAveragePooling2D(),

    # Reforma la salida para preparar la predicción de cajas delimitadoras
    Reshape((1, 1, 1280)),

    # Añade una capa convolucional para predicciones de cajas delimitadoras y probabilidades de clase
    Conv2D(4 * (4 + 1 + num_classes), kernel_size=3, padding='same')
])

# Compila el modelo (ajusta el optimizador y la función de pérdida según sea necesario)
modelo.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Entrena el modelo en el conjunto de datos de entrenamiento, validando en el conjunto de datos de validación
historia = modelo.fit(train_ds, epochs=100, validation_data=validation_ds)

# Muestra el resumen del modelo
modelo.summary()

# Define una función para trazar el historial de entrenamiento
def plot_training_history(historia):
    plt.figure(figsize=(12, 4))

    # Grafica los valores de precisión de entrenamiento y validación
    plt.subplot(1, 2, 1)
    plt.plot(historia.history['accuracy'])
    plt.plot(historia.history['val_accuracy'])
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
