import cv2
import numpy as np
import os
import re
import csv
try:
    # For newer TensorFlow installations (like on your PC)
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator,
        img_to_array,
        array_to_img,
        load_img,
    )
except ImportError:
    # For systems with standalone keras installed (like your Ubuntu workstation)
    from keras.preprocessing.image import (
        ImageDataGenerator,
        img_to_array,
        array_to_img,
        load_img,
    )


# Función para realizar aumento de datos
def procesar_imagen(ruta_imagen, carpeta_salida, archivo):
    img = load_img(ruta_imagen)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Parámetros de aumento de datos más completos
    datagen = ImageDataGenerator(
        preprocessing_function=apply_augmentations,
        rotation_range=3,        # Pequeña rotación (hasta 3 grados)
        width_shift_range=0.02,  # Desplazamiento horizontal
        height_shift_range=0.02, # Desplazamiento vertical
        zoom_range=0.05,         # Zoom aleatorio
        fill_mode='nearest'      # Método para rellenar píxeles creados por la transformación
    )

    # Generar imágenes aumentadas y guardarlas en la carpeta de salida
    nombres_archivos_aumentados = []
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=carpeta_salida, save_prefix=f'{os.path.splitext(archivo)[0]}_aug', save_format='jpg'):
        i += 1
        nombre_archivo_aumentado = f'{os.path.splitext(archivo)[0]}_aug_{i}.jpg'
        nombres_archivos_aumentados.append(nombre_archivo_aumentado)
        if i >= 10:  # Aumentado de 7 a 10 imágenes aumentadas por imagen de entrada
            break
    return nombres_archivos_aumentados

def apply_augmentations(image):
    # Convertir a escala de grises
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Aplicar desenfoque (ligeramente aumentado)
    blur_amount = np.random.uniform(0, 5.5)  # Aumentado de 4.6 a 5.5
    image = cv2.GaussianBlur(image, (5, 5), blur_amount)

    # Añadir ruido gaussiano (aumentado)
    noise_amount = np.random.uniform(0.005, 0.035)  # Aumentado de 0-0.0199 a 0.005-0.035
    noise = np.random.normal(0, noise_amount * 255, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Añadir ocasionalmente ruido de sal y pimienta
    if np.random.random() < 0.3:  # 30% de probabilidad
        s_vs_p = 0.5
        amount = np.random.uniform(0.001, 0.004)
        # Sal
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        image[coords[0], coords[1], :] = 255
        # Pimienta
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        image[coords[0], coords[1], :] = 0
    
    # Añadir ocasionalmente ruido de disparo (shot noise)
    if np.random.random() < 0.25:  # 25% de probabilidad
        shot_noise = np.random.poisson(image * 0.1) / 0.1
        image = np.clip(shot_noise, 0, 255).astype(np.uint8)

    # Ajustar tono
    hue_shift = np.random.uniform(-14, 14)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:, :, 0] = (image[:, :, 0] + hue_shift) % 180
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    # Ajustar exposición (ligeramente aumentado)
    exposure_shift = np.random.uniform(-0.22, 0.22)  # Aumentado de ±0.18 a ±0.22
    image = np.clip(image * (1 + exposure_shift), 0, 255).astype(np.uint8)
    
    # Añadir ocasionalmente variación de contraste
    if np.random.random() < 0.4:  # 40% de probabilidad
        contrast_factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

    return image

def generar_anotaciones_csv(carpeta_salida, archivo_csv_salida):
    # Crear la carpeta de anotaciones si no existe
    carpeta_anotaciones = os.path.dirname(archivo_csv_salida)
    if not os.path.exists(carpeta_anotaciones):
        os.makedirs(carpeta_anotaciones)

    with open(archivo_csv_salida, mode="w", newline="", encoding="utf-8") as csv_out:
        writer = csv.writer(csv_out)
        for archivo in os.listdir(carpeta_salida):
            if archivo.endswith(".jpg") or archivo.endswith(".png"):
                ruta_imagen = os.path.join(carpeta_salida, archivo)
                # Obtener la primera letra del nombre del archivo
                nombre_base = os.path.splitext(archivo)[
                    0
                ]  # Obtiene el nombre sin extensión
                primera_letra = nombre_base[0]  # Toma el primer carácter
                # Verifica que sea una letra y la convierte a mayúscula
                if primera_letra.isalpha():
                    etiqueta = primera_letra.upper()
                    fila = [ruta_imagen, etiqueta, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
                    writer.writerow(fila)
                    print(f"Archivo: {archivo}, Etiqueta detectada: {etiqueta}")
                else:
                    print(f"No se pudo detectar etiqueta válida para: {archivo}")

def procesar_dataset(carpeta_entrada, carpeta_salida, archivo_csv_salida):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            ruta_entrada = os.path.join(carpeta_entrada, archivo)
            procesar_imagen(ruta_entrada, carpeta_salida, archivo)
            print(f"Procesado: {archivo}")
    
    # Generar archivo CSV con anotaciones
    generar_anotaciones_csv(carpeta_salida, archivo_csv_salida)
    print(f"Archivo CSV de anotaciones generado: {archivo_csv_salida}")

# Lista de rutas raíz
root_paths = [
    #SEPARADO
    "datasets/libroCorazonDelator"
]

# Ejecutar la función para cada ruta raíz
for root in root_paths:
    carpeta_de_entrada = root + "/separadas"
    carpeta_de_salida = root + "/aumentoDatosSeparadas"
    archivo_csv_salida = root + "/aumentoDatosSeparadas/anotaciones/anotaciones.csv"
    procesar_dataset(carpeta_de_entrada, carpeta_de_salida, archivo_csv_salida)