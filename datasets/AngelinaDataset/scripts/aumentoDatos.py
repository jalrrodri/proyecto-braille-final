import cv2
import numpy as np
import os
import re
import csv
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

# Mapeo de letras Braille a cuadrantes en relieve
braille_map = {
    'a': [1], 
    'b': [1, 3], 
    'c': [1, 2], 
    'd': [1, 2, 4], 
    'e': [1, 4], 
    'f': [1, 2, 3], 
    'g': [1, 2, 3, 4], 
    'h': [1, 3, 4], 
    'i': [2, 3], 
    'j': [2, 3, 4], 
    'k': [1, 5], 
    'l': [1, 3, 5], 
    'm': [1, 2, 5], 
    'n': [1, 2, 4, 5], 
    'o': [1, 4, 5], 
    'p': [1, 2, 3, 5], 
    'q': [1, 2, 3, 4, 5], 
    'r': [1, 3, 4, 5], 
    's': [2, 3, 5], 
    't': [2, 3, 4, 5], 
    'u': [1, 5, 6], 
    'v': [1, 3, 5, 6], 
    'w': [2, 3, 4, 6], 
    'x': [1, 2, 5, 6], 
    'y': [1, 2, 4, 5, 6], 
    'z': [1, 4, 5, 6]
}

# Función para realizar aumento de datos
def procesar_imagen(ruta_imagen, carpeta_salida, archivo):
    img = load_img(ruta_imagen)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Parámetros de aumento de datos
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=7,
        brightness_range=[0.8, 1.2],
        zoom_range=[0.8, 1.2]
    )

    # Generar imágenes aumentadas
    i = 0
    nombres_archivos_aumentados = []
    for batch in datagen.flow(x, batch_size=1, save_to_dir=carpeta_salida, save_prefix=f'{os.path.splitext(archivo)[0]}_aug', save_format='jpg'):
        i += 1
        nombre_archivo_aumentado = f'{os.path.splitext(archivo)[0]}_aug_{i}.jpg'
        nombres_archivos_aumentados.append(nombre_archivo_aumentado)
        if i >= 5:  # Generar 5 imágenes aumentadas por imagen de entrada
            break
    return nombres_archivos_aumentados

def procesar_dataset(carpeta_entrada, carpeta_salida, carpeta_anotaciones_entrada, carpeta_anotaciones_salida):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    if not os.path.exists(carpeta_anotaciones_salida):
        os.makedirs(carpeta_anotaciones_salida)
    
    mapa_archivos_aumentados = {}
    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            ruta_entrada = os.path.join(carpeta_entrada, archivo)
            nombres_archivos_aumentados = procesar_imagen(ruta_entrada, carpeta_salida, archivo)
            mapa_archivos_aumentados[archivo] = nombres_archivos_aumentados
            print(f"Procesado: {archivo}")
    
    # Copiar y actualizar anotaciones
    for archivo_csv in os.listdir(carpeta_anotaciones_entrada):
        if archivo_csv.endswith(".csv"):
            ruta_csv_entrada = os.path.join(carpeta_anotaciones_entrada, archivo_csv)
            ruta_csv_salida = os.path.join(carpeta_anotaciones_salida, archivo_csv)
            
            with open(ruta_csv_entrada, mode='r', newline='', encoding='utf-8') as csv_in, \
                 open(ruta_csv_salida, mode='w', newline='', encoding='utf-8') as csv_out:
                
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)
                
                for row in reader:
                    nombre_archivo_original = os.path.basename(row[0])
                    if nombre_archivo_original in mapa_archivos_aumentados:
                        for nombre_archivo_aumentado in mapa_archivos_aumentados[nombre_archivo_original]:
                            nueva_fila = row.copy()
                            nueva_fila[0] = os.path.join(carpeta_salida, nombre_archivo_aumentado)
                            writer.writerow(nueva_fila)
                    else:
                        writer.writerow(row)
    
    print("Procesamiento completado.")

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
    carpeta_anotaciones_entrada = root + "/traducido/filtros/anotaciones"
    carpeta_anotaciones_salida = root + "/traducido/aumentoDatos/anotaciones"
    procesar_dataset(carpeta_de_entrada, carpeta_de_salida, carpeta_anotaciones_entrada, carpeta_anotaciones_salida)