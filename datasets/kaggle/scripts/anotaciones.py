import os
import csv

# Directorio que contiene las fotos
photos_dir = 'datasets/kaggle'

# Archivo CSV de salida
output_csv_path = 'datasets/kaggle/anotaciones/kaggle.csv'

# Crear el directorio de salida si no existe
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Lista para almacenar las filas del CSV
rows = []

# Recorrer todos los archivos en el directorio de fotos
for filename in os.listdir(photos_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Ruta completa de la foto
        photo_path = os.path.join(photos_dir, filename).replace("\\", "/")
        
        # Etiqueta: primera letra del nombre del archivo
        label = filename[0].upper()
        
        # Añadir la fila al CSV
        row = [photo_path, label, '0.0', '0.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0']
        rows.append(row)

# Escribir las filas en el archivo CSV
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(rows)

print(f"Anotaciones guardadas en: {output_csv_path}")