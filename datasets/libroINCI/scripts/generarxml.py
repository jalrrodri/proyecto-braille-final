import os
from PIL import Image
import xml.etree.ElementTree as ET

def get_image_details(image_path):
    """
    Obtiene detalles de la imagen como ancho, alto y profundidad.

    :param image_path: Ruta al archivo de imagen.
    :return: Tupla que contiene el ancho, alto y profundidad de la imagen.
    """
    try:
        image = Image.open(image_path)
        width, height = image.size
        depth = len(image.getbands())  # Número de canales (por ejemplo, 3 para RGB)
        return width, height, depth
    except Exception as e:
        print(f"Error al obtener detalles de la imagen {image_path}: {e}")
        return None, None, None

def create_unified_pascal_voc_xml(image_folder, output_file):
    """
    Crea un único archivo de anotación XML Pascal VOC para todas las imágenes en una carpeta.

    :param image_folder: Ruta a la carpeta que contiene las imágenes.
    :param output_file: Ruta para guardar el archivo XML unificado.
    """
    # Crear elemento raíz
    dataset = ET.Element("dataset")

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, filename)
            print(f"Procesando archivo: {image_path}")

            # Obtener detalles de la imagen
            width, height, depth = get_image_details(image_path)
            if width is None or height is None or depth is None:
                print(f"Omitiendo archivo debido a error: {image_path}")
                continue

            # Extraer etiqueta de la primera letra del nombre del archivo
            label = filename[0].lower()

            # Crear entrada para esta imagen
            image_elem = ET.SubElement(dataset, "image")

            filename_elem = ET.SubElement(image_elem, "filename")
            filename_elem.text = filename

            size_elem = ET.SubElement(image_elem, "size")
            width_elem = ET.SubElement(size_elem, "width")
            width_elem.text = str(width)
            height_elem = ET.SubElement(size_elem, "height")
            height_elem.text = str(height)
            depth_elem = ET.SubElement(size_elem, "depth")
            depth_elem.text = str(depth)

            object_elem = ET.SubElement(image_elem, "object")
            name_elem = ET.SubElement(object_elem, "name")
            name_elem.text = label

            bndbox_elem = ET.SubElement(object_elem, "bndbox")
            xmin_elem = ET.SubElement(bndbox_elem, "xmin")
            xmin_elem.text = "0"
            ymin_elem = ET.SubElement(bndbox_elem, "ymin")
            ymin_elem.text = "0"
            xmax_elem = ET.SubElement(bndbox_elem, "xmax")
            xmax_elem.text = str(width)
            ymax_elem = ET.SubElement(bndbox_elem, "ymax")
            ymax_elem.text = str(height)

    # Escribir XML en archivo
    try:
        tree = ET.ElementTree(dataset)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print(f"Archivo XML escrito exitosamente en {output_file}")
    except Exception as e:
        print(f"Error al escribir el archivo XML en {output_file}: {e}")

# Rutas de entrada y salida
image_folder = "./datasetprueba1"
output_folder = "./anotaciones/datasetprueba1.xml"
create_unified_pascal_voc_xml(image_folder, output_folder)




