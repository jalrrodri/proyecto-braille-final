import os
import csv
import cv2
import numpy as np
from pathlib import Path


def ensure_forward_slashes(path):
    """Asegura que la ruta utilice barras diagonales hacia adelante (/)"""
    return path.replace("\\", "/")


def create_directory_if_not_exists(directory):
    """Crea un directorio si no existe"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def crop_and_save_images(csv_path, output_base_dir):
    """
    Lee el CSV, recorta las imágenes según las coordenadas y guarda un nuevo CSV
    con las rutas actualizadas y coordenadas que abarcan toda la imagen.
    """
    # Crear directorios de salida
    output_images_dir = os.path.join(output_base_dir, "separadas")
    output_annotations_dir = os.path.join(output_images_dir, "anotaciones")

    create_directory_if_not_exists(output_images_dir)
    create_directory_if_not_exists(output_annotations_dir)

    # Preparar el archivo CSV de salida
    output_csv_path = os.path.join(output_annotations_dir, "anotaciones_recortadas.csv")

    # Diccionario para almacenar las nuevas filas del CSV
    new_csv_rows = []

    # Leer el CSV de entrada
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            if len(row) < 9:  # Verificar que la fila tenga suficientes columnas
                print(f"Advertencia: fila con formato incorrecto: {row}")
                continue

            # Extraer información de la fila
            image_path = row[0].replace(
                "\\", "/"
            )  # Normalizar a barras diagonales hacia adelante
            label = row[1]

            # Extraer coordenadas normalizadas del bounding box
            # Formato: x1, y1, x2, y2, x3, y3, x4, y4
            try:
                x1 = float(row[2])
                y1 = float(row[3])
                x3 = float(row[6])
                y3 = float(row[7])
            except ValueError as e:
                print(f"Error al convertir coordenadas para {image_path}: {e}")
                continue

            # Verificar si la imagen existe
            if not os.path.exists(image_path.replace("/", os.sep)):
                print(f"Advertencia: no se puede encontrar la imagen: {image_path}")
                continue

            # Cargar la imagen
            try:
                image = cv2.imread(image_path.replace("/", os.sep))
                if image is None:
                    print(f"Error al cargar la imagen: {image_path}")
                    continue

                h, w = image.shape[:2]

                # Convertir coordenadas normalizadas a píxeles
                x1_px = int(x1 * w)
                y1_px = int(y1 * h)
                x3_px = int(x3 * w)
                y3_px = int(y3 * h)

                # Recortar la imagen
                cropped_image = image[y1_px:y3_px, x1_px:x3_px]

                if cropped_image.size == 0:
                    print(f"Advertencia: recorte vacío para {image_path}")
                    continue

                # Generar nombre de archivo de salida
                image_filename = os.path.basename(image_path)
                base_name, ext = os.path.splitext(image_filename)
                output_filename = f"{label}_{base_name}{ext}"
                output_path = os.path.join(output_images_dir, output_filename).replace(
                    "\\", "/"
                )

                # Guardar la imagen recortada
                cv2.imwrite(output_path.replace("/", os.sep), cropped_image)

                # Crear nueva fila para el CSV con coordenadas que abarcan toda la imagen
                new_row = [
                    output_path,  # Nueva ubicación con barras diagonales hacia adelante
                    label,  # Etiqueta
                    "0.0",  # x1
                    "0.0",  # y1
                    "1.0",  # x2
                    "0.0",  # y2
                    "1.0",  # x3
                    "1.0",  # y3
                    "0.0",  # x4
                    "1.0",  # y4
                ]

                new_csv_rows.append(new_row)
                print(f"Recortada y guardada: {output_path}")

            except Exception as e:
                print(f"Error al procesar {image_path}: {e}")

    # Guardar el nuevo CSV
    with open(output_csv_path.replace("/", os.sep), "w", newline="") as output_csv:
        csv_writer = csv.writer(output_csv)
        csv_writer.writerows(new_csv_rows)

    print(f"\nProceso completado. CSV de salida guardado en: {output_csv_path}")
    print(f"Imágenes recortadas guardadas en: {output_images_dir}")
    print(f"Total de imágenes procesadas: {len(new_csv_rows)}")


if __name__ == "__main__":
    # Configuración - Cambia estas rutas según tu estructura de directorios
    input_csv_path = (
        "datasets/libroCorazonDelator/completas/anotaciones/csv/anotaciones.csv"
    )
    output_directory = (
        "datasets/libroCorazonDelator"  # Ejemplo basado en tu salida
    )  # Ejemplo basado en tu salida

    # Ejecutar el procesamiento
    crop_and_save_images(input_csv_path, output_directory)
