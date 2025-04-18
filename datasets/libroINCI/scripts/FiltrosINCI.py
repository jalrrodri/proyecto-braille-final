import cv2
import os
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
            # Filtros más suaves para cuadrantes en relieve
            
            # Mejora de contraste moderada usando CLAHE
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            roi = clahe.apply(roi)
            
            # Reducción de ruido suave
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            
            # Umbral adaptativo más moderado
            blockSize = 101  # Reducido de 1001 para ser menos agresivo
            C = 8  # Reducido de 15 para ser menos agresivo
            roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, blockSize, C)
        else:
            # Filtros mínimos para cuadrantes sin relieve
            
            # Suavizado ligero
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            
            # Umbral adaptativo ligero
            blockSize = 21  # Aumentado de 3 para ser menos agresivo
            C = 12  # Reducido de 100 para ser menos extremo
            roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, blockSize, C)
            
        img_procesada[y1:y2, x1:x2] = roi
    
    return img_procesada

def procesar_dataset(carpeta_entrada, carpeta_salida, archivo_csv_salida):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            letra = archivo[0].lower()
            ruta_entrada = os.path.join(carpeta_entrada, archivo)
            imagen_procesada = procesar_imagen(ruta_entrada, letra)
            if imagen_procesada is not None:
                ruta_salida = os.path.join(carpeta_salida, archivo)
                cv2.imwrite(ruta_salida, imagen_procesada)
                print(f"Procesado: {archivo}")
    
    # Generar archivo CSV con anotaciones
    generar_anotaciones_csv(carpeta_salida, archivo_csv_salida)
    print(f"Archivo CSV de anotaciones generado: {archivo_csv_salida}")

def generar_anotaciones_csv(carpeta_salida, archivo_csv_salida):
    # Crear la carpeta de anotaciones si no existe
    carpeta_anotaciones = os.path.dirname(archivo_csv_salida)
    if not os.path.exists(carpeta_anotaciones):
        os.makedirs(carpeta_anotaciones)

    with open(archivo_csv_salida, mode='w', newline='', encoding='utf-8') as csv_out:
        writer = csv.writer(csv_out)
        for archivo in os.listdir(carpeta_salida):
            if archivo.endswith(".jpg") or archivo.endswith(".png"):
                ruta_imagen = os.path.join(carpeta_salida, archivo)
                etiqueta = os.path.basename(archivo)[0]  # Obtener la primera letra del nombre del archivo
                fila = [ruta_imagen, etiqueta, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
                writer.writerow(fila)

carpeta_de_entrada = "datasets/libroINCI/datasetprueba1"
carpeta_de_salida = "datasets/libroINCI/datasetprueba1FILTROS"
archivo_csv_salida = "datasets/libroINCI/datasetprueba1FILTROS/anotaciones/datasetprueba1FILTROS.csv"
procesar_dataset(carpeta_de_entrada, carpeta_de_salida, archivo_csv_salida)