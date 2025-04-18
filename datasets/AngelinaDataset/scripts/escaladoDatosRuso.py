import cv2
import numpy as np
import os
import re
import csv
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
    # Crear carpeta si no existe
    archivo_csv_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(archivo_csv_salida, mode="w", newline="", encoding="utf-8") as csv_out:
        writer = csv.writer(csv_out)

        # # Escribir encabezado
        # writer.writerow(['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])

        for archivo in carpeta_salida.glob("*_resize.jp*g"):
            nombre_archivo = archivo.name

            # Extraer etiqueta usando el patrón labeled_X_
            match = re.search(r"labeled_([A-Za-z0-9])_", nombre_archivo)
            etiqueta = match.group(1).upper() if match else None

            if etiqueta:
                print(f"Archivo: {nombre_archivo}, Etiqueta detectada: {etiqueta}")

                # Bounding box normalizado (centrado en la imagen)
                fila = [str(archivo), etiqueta, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
                writer.writerow(fila)
            else:
                print(f"No se pudo detectar etiqueta para: {nombre_archivo}")

    print(f"Anotaciones guardadas en {archivo_csv_salida}")


def procesar_dataset(root_path):
    root_path = Path(root_path)

    # Definir las carpetas de entrada y salida
    subcarpetas = ["separado", "filtros", "aumentoDatos"]

    for subcarpeta in subcarpetas:
        carpeta_entrada = root_path / "traducido" / subcarpeta
        carpeta_salida = root_path / "traducido" / "redimensionado" / subcarpeta
        archivo_csv_salida = carpeta_salida / "anotaciones/anotaciones.csv"

        if not carpeta_entrada.exists():
            print(f"La carpeta {carpeta_entrada} no existe. Continuando...")
            continue

        # Crear carpeta de salida si no existe
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        # Procesar imágenes
        for archivo in carpeta_entrada.glob("*.jp*g"):
            redimensionar_imagen(archivo, carpeta_salida, archivo.name)
            print(f"Procesado: {archivo.name}")

        # Generar anotaciones
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
    "datasets/AngelinaDataset/handwritten/uploaded",
]

# Ejecutar la función para cada ruta raíz
for root in root_paths:
    print(f"\nProcesando {root}...")
    procesar_dataset(root)
