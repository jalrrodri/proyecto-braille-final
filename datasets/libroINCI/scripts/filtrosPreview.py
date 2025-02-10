import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar la imagen en escala de grises
image_path = "datasets/libroINCI/datasetprueba1/a.jpg"  # Asegúrate de que esta ruta sea correcta y apunte a un archivo de imagen específico
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se ha cargado correctamente
if image is None:
    print(f"Error: No se pudo cargar la imagen desde la ruta: {image_path}")
else:
    # Ajuste de contraste usando CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(image)

    # Reducción de ruido usando filtro Gaussiano
    denoised = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)

    # Aplicar umbral adaptativo para resaltar los puntos
    blockSize = 15  # Debe ser un número impar
    C = 5  # Constante que se resta de la media o la media ponderada
    thresholded_adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C
    )

    # Alternativa: Umbral global simple
    _, thresholded_global = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Mostrar los resultados
    titles = ['Original', 'Ajuste de Contraste', 'Reducción de Ruido', 'Umbral Adaptativo', 'Umbral Global']
    images = [image, contrast_enhanced, denoised, thresholded_adaptive, thresholded_global]

    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Guardar la imagen procesada
    output_path = "datasets/libroINCI/imagenPrueba.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, thresholded_adaptive)
    print(f"Imagen procesada guardada en: {output_path}")
