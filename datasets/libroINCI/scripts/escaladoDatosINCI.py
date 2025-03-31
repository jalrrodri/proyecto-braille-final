import cv2
import numpy as np
import os
import re
import csv

import cv2
import os
import csv
import re
import numpy as np
from pathlib import Path


def redimensionar_imagen(ruta_imagen, carpeta_salida, archivo):
    # Leer la imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"Error al leer {ruta_imagen}")
        return None

    # Solo definimos altura objetivo
    TARGET_HEIGHT = 512

    # Obtener dimensiones originales
    alto, ancho = img.shape[:2]

    # Calcular escala basada solo en la altura
    escala = TARGET_HEIGHT / alto

    # Calcular nuevo ancho manteniendo la relación de aspecto
    nuevo_ancho = int(ancho * escala)
    nuevo_alto = TARGET_HEIGHT

    # Redimensionar manteniendo la relación de aspecto
    imagen_final = cv2.resize(
        img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LANCZOS4
    )

    # Guardar imagen
    nombre_base = os.path.splitext(archivo)[0]
    ruta_salida = os.path.join(carpeta_salida, f"{nombre_base}_resize.jpg")
    cv2.imwrite(ruta_salida, imagen_final, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return ruta_salida


def generar_anotaciones_csv(carpeta_salida, archivo_csv_salida):
    # Convert string paths to Path objects
    carpeta_salida = Path(carpeta_salida)
    archivo_csv_salida = Path(archivo_csv_salida)

    # Create directory if it doesn't exist
    archivo_csv_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(archivo_csv_salida, mode="w", newline="", encoding="utf-8") as csv_out:
        writer = csv.writer(csv_out)

        # # Write header
        # writer.writerow(['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])

        for archivo in carpeta_salida.glob("*_resize.jp*g"):
            nombre_archivo = archivo.name

            # Extract first letter of filename as label
            etiqueta = nombre_archivo[0].upper()

            print(f"Archivo: {nombre_archivo}, Etiqueta detectada: {etiqueta}")

            # Normalized bounding box (centered in image)
            fila = [str(archivo), etiqueta, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
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
    "datasets/libroINCI/datasetprueba1",
    "datasets/libroINCI/datasetprueba1AUMENTODATOS",
    "datasets/libroINCI/datasetprueba1FILTROS",
]

# Ejecutar la función para cada ruta raíz
for root in root_paths:
    carpeta_de_entrada = root
    carpeta_de_salida = root + "/redimensionado"
    archivo_csv_final = root + "/redimensionado/anotaciones/anotaciones.csv"

    if os.path.exists(carpeta_de_entrada):
        # Procesar las imágenes y generar anotaciones
        procesar_dataset(carpeta_de_entrada, carpeta_de_salida, archivo_csv_final)
        print(f"Procesamiento completado para: {root}")
        print(f"Archivo CSV generado: {archivo_csv_final}")
