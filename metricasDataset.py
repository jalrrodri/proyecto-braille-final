import numpy as np
from keras_preprocessing.image import load_img, img_to_array
import os
import cv2

# Define el directorio que contiene tus imágenes
image_dir = "dataset"

# Inicializa variables para acumular la suma y el conteo de píxeles
total_sum = np.zeros(3)  # Suponiendo imágenes en formato RGB, por lo tanto, 3 canales
total_count = 0

# Itera sobre cada imagen en el directorio
for filename in os.listdir(image_dir):
    # Carga la imagen
    image_path = os.path.join(image_dir, filename)
    img = load_img(image_path)
    
    # Convierte la imagen a un array NumPy
    img_array = img_to_array(img)
    
    # Acumula la suma y el conteo de los valores de los píxeles
    total_sum += np.sum(img_array, axis=(0, 1))
    total_count += img_array.shape[0] * img_array.shape[1]

# Calcula la media y la desviación estándar
mean = total_sum / total_count
variance = np.zeros(3)  # Inicializa la varianza
for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    img = load_img(image_path)
    img_array = img_to_array(img)
    variance += np.sum((img_array - mean) ** 2, axis=(0, 1))
std_dev = np.sqrt(variance / total_count)

print("Media:", mean)
print("Desviación Estándar:", std_dev)

def obtener_dimensiones_entrada(image_dir):
    dimensiones_entrada = []
    # Itera sobre todos los archivos en la carpeta
    for nombre_archivo in os.listdir(image_dir):
        # Verifica si el archivo es una imagen (con extensiones .jpg, .png o .jpeg)
        if nombre_archivo.endswith('.jpg') or nombre_archivo.endswith('.png') or nombre_archivo.endswith('.jpeg'):
            ruta_imagen = os.path.join(image_dir, nombre_archivo)
            # Lee la imagen utilizando opencv
            img = cv2.imread(ruta_imagen)
            # Verifica si la imagen se ha cargado correctamente
            if img is not None:
                # Obtiene las dimensiones de la imagen
                alto, ancho, canales = img.shape
                dimensiones_entrada.append((alto, ancho, canales))
    return dimensiones_entrada

dimensiones_entrada = obtener_dimensiones_entrada(image_dir)
print("Dimensiones de entrada de las imágenes en la carpeta:", dimensiones_entrada)

