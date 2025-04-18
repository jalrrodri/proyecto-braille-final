import csv
from collections import defaultdict
import random

def igualar_etiquetas_anotaciones(input_csv_path, output_csv_path):
    # Leer el archivo CSV y contar la cantidad de cada valor en la segunda columna
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)
    
    # Contar la cantidad de cada valor en la segunda columna
    value_counts = defaultdict(int)
    for row in rows:
        value_counts[row[1]] += 1
    
    # Imprimir la cantidad de filas de cada valor en la segunda columna antes del filtro
    print("Cantidad de cada valor antes del filtro:")
    for value in sorted(value_counts):
        print(f"Valor: {value}, Cantidad: {value_counts[value]}")
    
    # Determinar el valor mínimo entre las cantidades de cada valor
    min_count = min(value_counts.values())
    
    # Filtrar las filas para que haya la misma cantidad de cada valor en la segunda columna
    filtered_rows = []
    value_counts_filtered = defaultdict(int)
    for row in rows:
        if value_counts_filtered[row[1]] < min_count:
            filtered_rows.append(row)
            value_counts_filtered[row[1]] += 1
    
    # Mezclar las filas filtradas de manera aleatoria
    random.shuffle(filtered_rows)
    
    # Dividir las filas filtradas en conjuntos de entrenamiento, validación y prueba
    train_split = int(0.8 * len(filtered_rows))
    val_split = int(0.9 * len(filtered_rows))
    
    train_rows = filtered_rows[:train_split]
    val_rows = filtered_rows[train_split:val_split]
    test_rows = filtered_rows[val_split:]
    
    # Agregar la columna de conjunto (TRAIN, VAL, TEST) como la primera columna
    for row in train_rows:
        row.insert(0, 'TRAIN')
    for row in val_rows:
        row.insert(0, 'VAL')
    for row in test_rows:
        row.insert(0, 'TEST')
    
    # Escribir las filas filtradas y divididas en el archivo CSV de salida
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(train_rows + val_rows + test_rows)
    
    # Imprimir la cantidad de filas de cada valor en la segunda columna después del filtro
    print("Cantidad de cada valor después del filtro:")
    for value in sorted(value_counts_filtered):
        print(f"Valor: {value}, Cantidad: {value_counts_filtered[value]}")
    
    print(f"Archivo CSV procesado y guardado en: {output_csv_path}")

# Rutas de los archivos CSV de entrada y salida
input_csv_path = 'anotaciones.csv'
output_csv_path = 'anotacionesIgualadas.csv'

igualar_etiquetas_anotaciones(input_csv_path, output_csv_path)