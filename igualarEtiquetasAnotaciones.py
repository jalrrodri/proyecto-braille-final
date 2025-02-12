import csv
from collections import defaultdict

def igualar_etiquetas_anotaciones(input_csv_path, output_csv_path):
    # Leer el archivo CSV y contar la cantidad de cada valor en la tercera columna
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)
    
    # Contar la cantidad de cada valor en la tercera columna
    value_counts = defaultdict(int)
    for row in rows:
        value_counts[row[2]] += 1
    
    # Determinar el valor mínimo entre las cantidades de cada valor
    min_count = min(value_counts.values())
    
    # Filtrar las filas para que haya la misma cantidad de cada valor en la tercera columna
    filtered_rows = []
    value_counts_filtered = defaultdict(int)
    for row in rows:
        if value_counts_filtered[row[2]] < min_count:
            filtered_rows.append(row)
            value_counts_filtered[row[2]] += 1
    
    # Escribir las filas filtradas en el archivo CSV de salida
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(filtered_rows)
    
    print(f"Archivo CSV procesado y guardado en: {output_csv_path}")

# Rutas de los archivos CSV de entrada y salida
input_csv_path = 'anotaciones.csv'
output_csv_path = 'anotacionesIgualadas.csv'

igualar_etiquetas_anotaciones(input_csv_path, output_csv_path)