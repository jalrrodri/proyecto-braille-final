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


def redimensionar_imagen(ruta_imagen, carpeta_salida, archivo):
    # Leer la imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"Error al leer {ruta_imagen}")
        return None

    # Tamaño cuadrado objetivo (ej. 512x512 para efficientdet_lite3)
    nuevo_tamano = 512

    # Obtener dimensiones originales
    alto, ancho = img.shape[:2]

    # Escalar manteniendo proporción
    escala = nuevo_tamano / max(alto, ancho)
    nuevo_ancho = int(ancho * escala)
    nuevo_alto = int(alto * escala)

    # Redimensionar sin distorsión
    imagen_redimensionada = cv2.resize(
        img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LANCZOS4
    )

    # Agregar padding para hacerla cuadrada
    delta_ancho = nuevo_tamano - nuevo_ancho
    delta_alto = nuevo_tamano - nuevo_alto
    top, bottom = delta_alto // 2, delta_alto - (delta_alto // 2)
    left, right = delta_ancho // 2, delta_ancho - (delta_ancho // 2)

    imagen_cuadrada = cv2.copyMakeBorder(
        imagen_redimensionada,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[128, 128, 128],
    )

    # Guardar imagen
    nombre_base = os.path.splitext(archivo)[0]
    ruta_salida = os.path.join(carpeta_salida, f"{nombre_base}_resize.jpg")
    cv2.imwrite(ruta_salida, imagen_cuadrada, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return ruta_salida


def generar_anotaciones_csv(carpeta_salida, archivo_csv_salida):
    # Crear carpeta si no existe
    carpeta_anotaciones = os.path.dirname(archivo_csv_salida)
    if not os.path.exists(carpeta_anotaciones):
        os.makedirs(carpeta_anotaciones)

    with open(archivo_csv_salida, mode="w", newline="", encoding="utf-8") as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(
            ["ruta", "etiqueta", "xmin", "ymin", "xmax", "ymax"]
        )  # Cabecera CSV

        for archivo in os.listdir(carpeta_salida):
            if archivo.endswith("_resize.jpg") or archivo.endswith("_resize.png"):
                ruta_imagen = os.path.join(carpeta_salida, archivo)

                # Extraer etiqueta de manera segura
                match = re.search(r"labeled_([A-Za-z0-9]+)_", archivo)
                etiqueta = match.group(1) if match else "Unknown"

                # Bounding box normalizado (asumiendo imagen cuadrada)
                fila = [ruta_imagen, etiqueta, 0.1, 0.1, 0.9, 0.9]
                writer.writerow(fila)

    print(f"Anotaciones guardadas en {archivo_csv_salida}")


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
# Subcarpetas a procesar
subcarpetas = ["/traducido/aumentoDatos", "/traducido/filtros", "/traducido/separado"]

# Ejecutar la función para cada ruta raíz y subcarpeta
for root in root_paths:
    # Crear una lista para almacenar todas las anotaciones
    todas_anotaciones = []

    for subcarpeta in subcarpetas:
        carpeta_de_entrada = root + subcarpeta
        carpeta_de_salida = root + "/traducido/redimensionado"
        archivo_csv_temporal = (
            root
            + f"/traducido/redimensionado/anotaciones/anotaciones_{subcarpeta.split('/')[-1]}.csv"
        )

        if os.path.exists(carpeta_de_entrada):
            # Procesar las imágenes
            procesar_dataset(
                carpeta_de_entrada, carpeta_de_salida, archivo_csv_temporal
            )

            # Leer las anotaciones generadas y agregarlas a la lista
            if os.path.exists(archivo_csv_temporal):
                with open(archivo_csv_temporal, mode="r", encoding="utf-8") as csv_in:
                    reader = csv.reader(csv_in)
                    todas_anotaciones.extend(list(reader))

                # Eliminar el archivo CSV temporal
                os.remove(archivo_csv_temporal)

    # Escribir todas las anotaciones en un único archivo CSV
    archivo_csv_final = (
        root + "/traducido/redimensionado/anotaciones/anotaciones_combinadas.csv"
    )
    carpeta_anotaciones = os.path.dirname(archivo_csv_final)
    if not os.path.exists(carpeta_anotaciones):
        os.makedirs(carpeta_anotaciones)

    with open(archivo_csv_final, mode="w", newline="", encoding="utf-8") as csv_out:
        writer = csv.writer(csv_out)
        writer.writerows(todas_anotaciones)

    print(f"Archivo CSV combinado generado: {archivo_csv_final}")