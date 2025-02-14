import pandas as pd
import os
import numpy as np

# /filtros/anotaciones
# /filtros/anotaciones
# Lista de directorios que contienen archivos CSV
csv_directories = [
    'datasets/AngelinaDataset/books/chudo_derevo_redmi/traducido/aumentoDatos/anotaciones',
    'datasets/AngelinaDataset/books/mdd_cannon1/traducido/aumentoDatos/anotaciones',
    'datasets/AngelinaDataset/books/mdd-redmi1/traducido/aumentoDatos/anotaciones',
    'datasets/AngelinaDataset/books/ola/traducido/aumentoDatos/anotaciones',
    'datasets/AngelinaDataset/books/skazki/traducido/aumentoDatos/anotaciones',
    'datasets/AngelinaDataset/books/telefon/traducido/aumentoDatos/anotaciones',
    'datasets/AngelinaDataset/books/uploaded/traducido/aumentoDatos/anotaciones',
    'datasets/AngelinaDataset/handwritten/ang_redmi/traducido/aumentoDatos/anotaciones',
    'datasets/AngelinaDataset/handwritten/kov/traducido/aumentoDatos/anotaciones',
    'datasets/AngelinaDataset/handwritten/uploaded/traducido/aumentoDatos/anotaciones'
    # Agrega más directorios según sea necesario
]

# Lista de rutas de archivos CSV individuales
csv_files = [
    'datasets/libroINCI/datasetprueba1FILTROS/anotaciones/datasetprueba1FILTROS.csv',
    'datasets/kaggle/anotaciones/kaggle.csv'
    # Agrega más archivos individuales según sea necesario
]

# Agregar todos los archivos CSV de los directorios a la lista csv_files
for directory in csv_directories:
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(directory, file))

# Crear una lista vacía para contener los dataframes
dataframes = []

# Leer cada archivo CSV y agregar el dataframe a la lista
for file in csv_files:
    df = pd.read_csv(file, header=None)
    dataframes.append(df)

# Concatenar todos los dataframes en uno solo
combined_df = pd.concat(dataframes, ignore_index=True)

# Crear una nueva columna con asignación aleatoria de 'TRAIN', 'VAL', 'TEST'
np.random.seed(42)  # Para reproducibilidad
labels = np.random.choice(['TRAIN', 'VAL', 'TEST'], size=len(combined_df), p=[0.8, 0.1, 0.1])
combined_df.insert(0, 'set', labels)

# Obtener el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definir la ruta del archivo de salida
output_file = os.path.join(script_dir, 'anotaciones.csv')

# Guardar el dataframe combinado en un archivo CSV
combined_df.to_csv(output_file, index=False, header=False)

print(f'Archivo CSV combinado guardado en {output_file}')