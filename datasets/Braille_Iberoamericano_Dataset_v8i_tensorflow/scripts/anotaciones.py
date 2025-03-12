import os
import pandas as pd

def process_csv(folder_name):
    base_path = "datasets/Braille_Iberoamericano_Dataset_v8i_tensorflow/"
    folder_path = os.path.join(base_path, folder_name)
    data_frames = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, header=0)  # Leer el archivo incluyendo el encabezado

            # Eliminar las columnas width y height
            df.drop(df.columns[[1, 2]], axis=1, inplace=True)

            # Eliminar las filas cuya columna class tenga valor 'enie'
            df = df[df.iloc[:, 1] != 'enie']

            # Poner los valores de la columna class en mayúscula
            df.iloc[:, 1] = df.iloc[:, 1].str.upper()

            # Actualizar la ruta de la primera columna
            df.iloc[:, 0] = base_path + folder_name + "/" + df.iloc[:, 0]

            # Añadir columnas vacías a partir de la tercera columna
            df.insert(4, 'empty1', '')
            df.insert(5, 'empty2', '')
            df.insert(8, 'empty3', '')
            df.insert(9, 'empty4', '')

            data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)

# Procesar las carpetas test, train y valid y concatenar los resultados
all_data_frames = []
for folder in ['test', 'train', 'valid']:
    all_data_frames.append(process_csv(folder))

final_df = pd.concat(all_data_frames, ignore_index=True)

# Guardar el CSV concatenado en la carpeta formato
output_folder = os.path.join("datasets/Braille_Iberoamericano_Dataset_v8i_tensorflow/", "formato")
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, "concatenated.csv")
final_df.to_csv(output_file_path, index=False)

# Eliminar la primera fila del CSV concatenado
# Para usar la primera fila de datos como nuevo encabezado
final_df = pd.read_csv(output_file_path, header=0)  # Lee el encabezado normal
new_header = final_df.iloc[0]  # Obtiene la primera fila de datos
final_df = final_df.iloc[1:]  # Elimina la primera fila de datos
final_df.columns = new_header  # Establece la primera fila como nuevo encabezado
final_df.to_csv(output_file_path, index=False)