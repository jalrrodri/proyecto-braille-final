import cv2
import numpy as np
import os
import re
import csv
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

# Función para realizar aumento de datos
def procesar_imagen(ruta_imagen, carpeta_salida, archivo):
    img = load_img(ruta_imagen)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Parámetros de aumento de datos
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=5,
        brightness_range=[0.8, 1.2],
        zoom_range=[0.8, 1.2]
    )

    # Generar imágenes aumentadas y guardarlas en la carpeta de salida
    nombres_archivos_aumentados = []
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=carpeta_salida, save_prefix=f'{os.path.splitext(archivo)[0]}_aug', save_format='jpg'):
        i += 1
        nombre_archivo_aumentado = f'{os.path.splitext(archivo)[0]}_aug_{i}.jpg'
        nombres_archivos_aumentados.append(nombre_archivo_aumentado)
        if i >= 7:  # Generar 5 imágenes aumentadas por imagen de entrada
            break
    return nombres_archivos_aumentados

def generar_anotaciones_csv(carpeta_salida, archivo_csv_salida):
    # Crear la carpeta de anotaciones si no existe
    carpeta_anotaciones = os.path.dirname(archivo_csv_salida)
    if not os.path.exists(carpeta_anotaciones):
        os.makedirs(carpeta_anotaciones)

    with open(archivo_csv_salida, mode='w', newline='', encoding='utf-8') as csv_out:
        writer = csv.writer(csv_out)
        for archivo in os.listdir(carpeta_salida):
            if archivo.endswith(".jpg") or archivo.endswith(".png"):
                ruta_imagen = os.path.join(carpeta_salida, archivo)
                etiqueta = re.search(r'labeled_([A-Z])_', archivo).group(1)
                fila = [ruta_imagen, etiqueta, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
                writer.writerow(fila)

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
    "datasets/AngelinaDataset/books/chudo_derevo_redmi",
    "datasets/AngelinaDataset/books/mdd_cannon1",
    "datasets/AngelinaDataset/books/mdd-redmi1",
    "datasets/AngelinaDataset/books/ola",
    "datasets/AngelinaDataset/books/skazki",
    "datasets/AngelinaDataset/books/telefon",
    "datasets/AngelinaDataset/books/uploaded",
    "datasets/AngelinaDataset/handwritten/ang_redmi",
    "datasets/AngelinaDataset/handwritten/kov",
    "datasets/AngelinaDataset/handwritten/uploaded"
]

# Ejecutar la función para cada ruta raíz
for root in root_paths:
    carpeta_de_entrada = root + "/traducido/filtros"
    carpeta_de_salida = root + "/traducido/aumentoDatos"
    archivo_csv_salida = root + "/traducido/aumentoDatos/anotaciones/anotaciones.csv"
    procesar_dataset(carpeta_de_entrada, carpeta_de_salida, archivo_csv_salida)