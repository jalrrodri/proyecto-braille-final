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
        # writer.writerow(
        #     ["ruta", "etiqueta", "xmin", "ymin", "xmax", "ymax"]
        # )  # Cabecera CSV

        for archivo in os.listdir(carpeta_salida):
            if archivo.endswith("_resize.jpg") or archivo.endswith("_resize.png"):
                ruta_imagen = os.path.join(carpeta_salida, archivo)

                # Extraer etiqueta de manera segura
                match = re.search(r"labeled_([A-Za-z0-9]+)_", archivo)
                etiqueta = match.group(1) if match else "Unknown"

                # Bounding box normalizado (asumiendo imagen cuadrada)
                fila = [ruta_imagen, etiqueta, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]
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
    "datasets/libroINCI/datasetprueba1",
    "datasets/libroINCI/datasetprueba1AUMENTODATOS",
    "datasets/libroINCI/datasetprueba1FILTROS"
]

# Ejecutar la función para cada ruta raíz
for root in root_paths:
    carpeta_de_entrada = root
    carpeta_de_salida = root + "/redimensionado"
    archivo_csv_final = root + "/traducido/redimensionado/anotaciones/anotaciones.csv"

    if os.path.exists(carpeta_de_entrada):
        # Procesar las imágenes y generar anotaciones
        procesar_dataset(carpeta_de_entrada, carpeta_de_salida, archivo_csv_final)
        print(f"Procesamiento completado para: {root}")
        print(f"Archivo CSV generado: {archivo_csv_final}")
