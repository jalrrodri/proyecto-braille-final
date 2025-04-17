import os
from collections import defaultdict
from pathlib import Path

def contar_imagenes_por_letra():
    # Definir la ruta de la carpeta
    ruta_carpeta = Path("datasets/libroCorazonDelator/separadas")
    
    # Crear un diccionario para almacenar el conteo
    conteo = defaultdict(int)
    
    # Verificar que la carpeta existe
    if not ruta_carpeta.exists():
        print(f"Error: La carpeta {ruta_carpeta.absolute()} no existe")
        return
    
    # Contar imágenes por letra
    for archivo in ruta_carpeta.glob("*.jp*g"):
        # Obtener la primera letra del nombre del archivo
        primera_letra = archivo.stem[0].upper()
        if primera_letra.isalpha():
            conteo[primera_letra] += 1
    
    # Ordenar el diccionario por letra
    conteo_ordenado = dict(sorted(conteo.items()))
    
    # Imprimir resultados
    print("\nConteo de imágenes por letra:")
    print("-" * 30)
    total_imagenes = 0
    for letra, cantidad in conteo_ordenado.items():
        print(f"Letra {letra}: {cantidad} imágenes")
        total_imagenes += cantidad
    
    print("-" * 30)
    print(f"Total de imágenes: {total_imagenes}")

if __name__ == "__main__":
    contar_imagenes_por_letra()