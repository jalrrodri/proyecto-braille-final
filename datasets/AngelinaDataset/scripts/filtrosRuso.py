import cv2
import os
import re
import csv

# Mapeo de letras Braille a cuadrantes en relieve
braille_map = {
    'a': [1], 
    'b': [1, 3], 
    'c': [1, 2], 
    'd': [1, 2, 4], 
    'e': [1, 4], 
    'f': [1, 2, 3], 
    'g': [1, 2, 3, 4], 
    'h': [1, 3, 4], 
    'i': [2, 3], 
    'j': [2, 3, 4], 
    'k': [1, 5], 
    'l': [1, 3, 5], 
    'm': [1, 2, 5], 
    'n': [1, 2, 4, 5], 
    'o': [1, 4, 5], 
    'p': [1, 2, 3, 5], 
    'q': [1, 2, 3, 4, 5], 
    'r': [1, 3, 4, 5], 
    's': [2, 3, 5], 
    't': [2, 3, 4, 5], 
    'u': [1, 5, 6], 
    'v': [1, 3, 5, 6], 
    'w': [2, 3, 4, 6], 
    'x': [1, 2, 5, 6], 
    'y': [1, 2, 4, 5, 6], 
    'z': [1, 4, 5, 6]
}

def procesar_imagen(ruta_imagen, letra):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"No se pudo cargar la imagen: {ruta_imagen}")
        return
    
    h, w = img.shape
    cuadrantes = [
        (0, h//3, 0, w//2),   # 1
        (0, h//3, w//2, w),   # 2
        (h//3, 2*h//3, 0, w//2),   # 3
        (h//3, 2*h//3, w//2, w),   # 4
        (2*h//3, h, 0, w//2),   # 5
        (2*h//3, h, w//2, w)   # 6
    ]
    
    # Obtener cuadrantes en relieve para la letra
    relieves = braille_map.get(letra, [])
    
    img_procesada = img.copy()
    for i, (y1, y2, x1, x2) in enumerate(cuadrantes, start=1):
        roi = img[y1:y2, x1:x2]
        if i in relieves:
            # Ajuste de contraste usando CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            roi = clahe.apply(roi)
            
            # Reducción de ruido usando filtro Gaussiano
            roi = cv2.GaussianBlur(roi, (5, 5), 0)
            
            # Umbral adaptativo
            blockSize = 1001  # Debe ser un número impar
            C = 15  # Constante que se resta de la media o la media ponderada
            roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
            
            # Alternativa: Umbral global simple
            _, roi_global = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Suavizar cuadrantes sin relieve
            roi = cv2.GaussianBlur(roi, (5, 5), 0)
            
            # Umbral adaptativo
            blockSize = 3  # Debe ser un número impar
            C = 100  # Constante que se resta de la media o la media ponderada
            roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
            
            # Alternativa: Umbral global simple
            _, roi_global = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_procesada[y1:y2, x1:x2] = roi
    
    return img_procesada

def procesar_dataset(root):
    carpeta_entrada = root + "/traducido/separado"
    carpeta_salida = root + "/traducido/filtros"
    carpeta_anotaciones_entrada = root + "/traducido/separado/anotaciones"
    carpeta_anotaciones_salida = root + "/traducido/filtros/anotaciones"

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    if not os.path.exists(carpeta_anotaciones_salida):
        os.makedirs(carpeta_anotaciones_salida)
    
    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            # Extraer la letra del nombre del archivo usando una expresión regular
            match = re.search(r'labeled_(\w)_', archivo)
            if match:
                letra = match.group(1).lower()
                ruta_entrada = os.path.join(carpeta_entrada, archivo)
                imagen_procesada = procesar_imagen(ruta_entrada, letra)
                if imagen_procesada is not None:
                    ruta_salida = os.path.join(carpeta_salida, archivo)
                    cv2.imwrite(ruta_salida, imagen_procesada)
                    print(f"Procesado: {archivo}")
    
    # Copiar y actualizar las anotaciones
    for archivo_csv in os.listdir(carpeta_anotaciones_entrada):
        if archivo_csv.endswith(".csv"):
            ruta_csv_entrada = os.path.join(carpeta_anotaciones_entrada, archivo_csv)
            ruta_csv_salida = os.path.join(carpeta_anotaciones_salida, archivo_csv)
            
            with open(ruta_csv_entrada, mode='r', newline='', encoding='utf-8') as csv_in, \
                 open(ruta_csv_salida, mode='w', newline='', encoding='utf-8') as csv_out:
                
                reader = csv.reader(csv_in)
                writer = csv.writer(csv_out)
                
                for row in reader:
                    # Actualizar la primera columna con la nueva ruta
                    row[0] = row[0].replace(carpeta_entrada, carpeta_salida)
                    writer.writerow(row)
    
    print(f"Procesamiento completado para {root}.")

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
    procesar_dataset(root)