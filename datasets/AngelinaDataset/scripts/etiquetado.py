import csv
import os
import re

def convert_number_to_letter(number):
    """
    Convierte un número a su letra correspondiente según la tabla de codificación dada.

    Args:
        number: El número a convertir.

    Returns:
        La letra correspondiente según la tabla de codificación.
    """
    encode_table = {
        0: ' ',
        1: 'A',
        2: '1',
        3: 'B',
        4: "'",
        5: 'K',
        6: '2',
        7: 'L',
        8: '3',
        9: 'C',
        10: 'I',
        11: 'F',
        12: '/',
        13: 'M',
        14: 'S',
        15: 'P',
        16: '"',
        17: 'E',
        18: '3',
        19: 'H',
        20: '9',
        21: 'O',
        22: '6',
        23: 'R',
        24: '^',
        25: 'D',
        26: 'J',
        27: 'G',
        28: '>',
        29: 'N',
        30: 'T',
        31: 'Q',
        32: ',',
        33: '*',
        34: '5',
        35: '-',
        36: '<',
        37: 'U',
        38: '8',
        39: 'V',
        40: '.',
        41: '%',
        42: '$',
        43: '+',
        44: 'X',
        45: '!',
        46: '&',
        47: ';',
        48: ':',
        49: '4',
        50: '0',
        51: '[',
        52: ']',
        53: '8',
        54: '(',
        55: ')',
        56: 'W',
        57: '7',
        58: '2',
        59: '1',
        60: 'Z',
        61: 'Y',
        62: '=',
        63: '6'
    }

    return encode_table.get(number, '')

def contains_only_letters(s):
    """
    Verifica si una cadena contiene solo letras.

    Args:
        s: La cadena a verificar.

    Returns:
        True si la cadena contiene solo letras, False en caso contrario.
    """
    return bool(re.match(r'^[A-Za-z]+$', s))

def generate_csv(folder_path, output_folder):
    """
    Genera archivos CSV a partir de una carpeta que contiene JPEGs y CSVs correspondientes, asignando etiquetas basadas en otro archivo CSV.

    Args:
        folder_path: Ruta a la carpeta que contiene JPEGs y CSVs.
        output_folder: Ruta a la carpeta donde se guardarán los archivos CSV de salida.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(folder_path, filename)
            image_path = filepath.replace("\\", "/")  # Reemplazar barras invertidas con barras diagonales

            # Extraer etiqueta y otros valores del archivo CSV correspondiente
            csv_filename = os.path.splitext(filename)[0] + ".csv"
            csv_filepath = os.path.join(folder_path, csv_filename)
            if not os.path.exists(csv_filepath):
                print(f"El archivo CSV {csv_filepath} no existe. Omitiendo.")
                continue

            data = []
            with open(csv_filepath, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=';')  # Especificar delimitador
                next(csvreader)  # Omitir fila de encabezado
                for row in csvreader:
                    if len(row) < 5:  # Asegurarse de que haya suficientes columnas en la fila
                        print(f"Omitiendo fila de {csv_filename} debido a columnas insuficientes.")
                        continue
                    label = convert_number_to_letter(int(row[4]))  # Convertir número a letra según la tabla de codificación
                    other_values = [row[i] for i in range(4)]  # Extraer las primeras cuatro columnas
                    data.append([image_path, label] + other_values)  # Agregar datos a la lista

            # Eliminar filas donde la etiqueta contiene números o caracteres especiales
            data = [row for row in data if contains_only_letters(row[1])]

            # Escribir datos en el archivo CSV
            output_filename = os.path.join(output_folder, os.path.splitext(filename)[0] + "_traducido.csv")
            with open(output_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # csvwriter.writerow(['Label', 'Image Path', '1st Column', '2nd Column', '3rd Column', '4th Column'])
                csvwriter.writerows(data)

            print(f"Archivo CSV generado exitosamente: {output_filename}")

# Lista de rutas de carpetas
folder_paths = [
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

# Ejecutar generate_csv para cada ruta de carpeta
for folder_path in folder_paths:
    output_folder = folder_path + "/traducido"
    generate_csv(folder_path, output_folder)
