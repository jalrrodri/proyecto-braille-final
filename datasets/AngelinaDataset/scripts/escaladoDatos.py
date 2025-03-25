import cv2
import numpy as np
import os
import re
import csv

def redimensionar_imagen(ruta_imagen, carpeta_salida, archivo):
    # Leer la imagen
    img = cv2.imread(ruta_imagen)
    
    # Calcular altura para mantener proporción 3:4
    nuevo_ancho = 512
    nuevo_alto = int(nuevo_ancho * 4 / 3)
    
    # Usar interpolación INTER_LANCZOS4 para mejor calidad
    imagen_redimensionada = cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LANCZOS4)
    
    # Generar ruta de salida
    nombre_base = os.path.splitext(archivo)[0]
    ruta_salida = os.path.join(carpeta_salida, f'{nombre_base}_resize.jpg')
    
    # Guardar imagen con alta calidad
    cv2.imwrite(ruta_salida, imagen_redimensionada, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return f'{nombre_base}_resize.jpg'

def generar_anotaciones_csv(carpeta_salida, archivo_csv_salida):
    # Crear la carpeta de anotaciones si no existe
    carpeta_anotaciones = os.path.dirname(archivo_csv_salida)
    if not os.path.exists(carpeta_anotaciones):
        os.makedirs(carpeta_anotaciones)

    with open(archivo_csv_salida, mode='w', newline='', encoding='utf-8') as csv_out:
        writer = csv.writer(csv_out)
        for archivo in os.listdir(carpeta_salida):
            if archivo.endswith("_resize.jpg") or archivo.endswith("_resize.png"):
                ruta_imagen = os.path.join(carpeta_salida, archivo)
                # Extraer etiqueta del nombre original del archivo
                etiqueta = re.search(r'labeled_([A-Z])_', archivo).group(1)
                fila = [ruta_imagen, etiqueta, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
                writer.writerow(fila)

def procesar_dataset(carpeta_entrada, carpeta_salida, archivo_csv_salida):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            ruta_entrada = os.path.join(carpeta_entrada, archivo)
            redimensionar_imagen(ruta_entrada, carpeta_salida, archivo)
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
    carpeta_de_entrada = root + "/traducido/aumentoDatos"
    carpeta_de_salida = root + "/traducido/redimensionado"
    archivo_csv_salida = root + "/traducido/redimensionado/anotaciones/anotaciones.csv"
    procesar_dataset(carpeta_de_entrada, carpeta_de_salida, archivo_csv_salida)