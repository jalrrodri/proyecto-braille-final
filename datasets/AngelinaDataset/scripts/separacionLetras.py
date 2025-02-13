import os
import csv
from PIL import Image

def procesar_separacion_letras(root):
    # --- Configuración ---
    input_csv_dir = root + '/traducido'  # Directorio que contiene los archivos CSV originales
    output_csv_dir = root + '/traducido/separado/anotaciones'  # Directorio para guardar los archivos CSV actualizados
    output_images_dir = root + '/traducido/separado'  # Directorio donde se guardarán las imágenes recortadas

    # Crear los directorios de salida si no existen
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)

    # Procesar cada archivo CSV en el directorio de entrada
    for csv_filename in os.listdir(input_csv_dir):
        if csv_filename.lower().endswith('.csv'):
            input_csv_path = os.path.join(input_csv_dir, csv_filename)
            
            # Agregar '_separados' al nombre del archivo CSV de salida
            name, ext = os.path.splitext(csv_filename)
            output_csv_filename = f"{name}_separados{ext}"
            output_csv_path = os.path.join(output_csv_dir, output_csv_filename)

            with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csv_in, \
                 open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_out:
                
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)
                
                # Si tu archivo CSV tiene un encabezado, descomenta las siguientes dos líneas:
                # header = next(reader)
                # writer.writerow(header)
                
                for row in reader:
                    # Omitir filas vacías
                    if not row:
                        continue
                    
                    # Estructura esperada de la fila:
                    # [image_path, label, left, top, right, bottom]
                    orig_image_path = row[0]
                    label = row[1]
                    
                    try:
                        # Convertir coordenadas normalizadas (asumidas en [0, 1)) a flotantes.
                        left_norm   = float(row[2])
                        top_norm    = float(row[3])
                        right_norm  = float(row[4])
                        bottom_norm = float(row[5])
                    except ValueError as ve:
                        print(f"Error al analizar las coordenadas en la fila {row}: {ve}")
                        continue
                    
                    # Abrir la imagen original
                    try:
                        image = Image.open(orig_image_path)
                    except Exception as e:
                        print(f"Error al abrir la imagen {orig_image_path}: {e}")
                        continue

                    # Obtener las dimensiones de la imagen
                    img_width, img_height = image.size
                    
                    # Convertir coordenadas normalizadas a coordenadas de píxeles.
                    # Asegurarse de que estos valores estén dentro de los límites de la imagen.
                    left_px   = int(left_norm * img_width)
                    top_px    = int(top_norm * img_height)
                    right_px  = int(right_norm * img_width)
                    bottom_px = int(bottom_norm * img_height)
                    
                    # Recortar la imagen. La caja es (left, top, right, bottom)
                    cropped_image = image.crop((left_px, top_px, right_px, bottom_px))
                    
                    # Generar un nuevo nombre de archivo para la imagen recortada.
                    # Este ejemplo usa el nombre base de la imagen original, agregando la etiqueta y las coordenadas de píxeles.
                    base_name = os.path.basename(orig_image_path)
                    name, ext = os.path.splitext(base_name)
                    new_filename = f"{name}_{label}_{left_px}_{top_px}_{right_px}_{bottom_px}{ext}"
                    new_image_path = os.path.join(output_images_dir, new_filename)
                    
                    # Guardar la imagen recortada.
                    try:
                        cropped_image.save(new_image_path)
                    except Exception as e:
                        print(f"Error al guardar la imagen recortada {new_image_path}: {e}")
                        continue
                    
                    # Actualizar la fila del CSV: reemplazar la ruta de la imagen original con la nueva ruta de la imagen recortada.
                    row[0] = new_image_path.replace("\\", "/")
                    
                    # Reemplazar las últimas 4 columnas con las 8 columnas especificadas
                    row = row[:2] + ['0.0', '0.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0']
                    
                    # Escribir la fila actualizada en el archivo CSV de salida.
                    writer.writerow(row)

    print(f"Procesamiento completado para {root}. Imágenes recortadas guardadas y CSVs actualizados.")

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
    procesar_separacion_letras(root)
