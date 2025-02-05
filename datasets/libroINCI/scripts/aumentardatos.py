import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Definir el directorio de origen y el directorio de destino
source_dir = "./datasetprueba1"
target_dir = "./datasetprueba1_augmented"

# Número de copias a realizar
num_copies = 1000

# Asegurarse de que el directorio de destino exista
os.makedirs(target_dir, exist_ok=True)

# Obtener todos los archivos en el directorio de origen
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Función para generar un nombre de archivo único
def generate_unique_name(base_name, ext, existing_files):
    counter = 1
    new_name = f"{base_name}_{counter}{ext}"
    while new_name in existing_files:
        counter += 1
        new_name = f"{base_name}_{counter}{ext}"
    return new_name

# Función para aumentar una imagen
def augment_image(image):
    # Rotación aleatoria
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        image = image.rotate(angle)

    # Traslación aleatoria
    if random.random() > 0.5:
        max_translate = 10
        x_translate = random.uniform(-max_translate, max_translate)
        y_translate = random.uniform(-max_translate, max_translate)
        image = image.transform(image.size, Image.AFFINE, (1, 0, x_translate, 0, 1, y_translate))

    # Escalado aleatorio
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Contraste aleatorio
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))

    # Brillo aleatorio
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))

    # Saturación aleatoria
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))

    # Tono aleatorio
    if random.random() > 0.5:
        image = np.array(image.convert('HSV'))
        image[..., 0] = (image[..., 0].astype(int) + random.randint(-10, 10)) % 256
        image = Image.fromarray(image, 'HSV').convert('RGB')

    # Desenfoque aleatorio
    if random.random() > 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2)))

    # Ruido aleatorio
    if random.random() > 0.5:
        noise = np.random.normal(0, 25, (image.height, image.width, 3)).astype(np.uint8)
        image = Image.fromarray(np.clip(np.array(image) + noise, 0, 255).astype(np.uint8))

    return image

# Replicar el conjunto de datos
existing_files = set(os.listdir(target_dir))
for _ in range(num_copies):
    for file in files:
        source_path = os.path.join(source_dir, file)
        base_name, ext = os.path.splitext(file)
        new_name = generate_unique_name(base_name, ext, existing_files)
        target_path = os.path.join(target_dir, new_name)

        # Abrir la imagen y aplicar aumento
        image = Image.open(source_path)
        augmented_image = augment_image(image)
        augmented_image.save(target_path)

        existing_files.add(new_name)

print(f"Conjunto de datos replicado y aumentado a {num_copies} imágenes en {target_dir}.")
