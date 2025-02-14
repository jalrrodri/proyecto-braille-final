import os
import shutil

def eliminar_contenido_aumentoDatos(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == 'aumentoDatos':
                carpeta_aumentoDatos = os.path.join(dirpath, dirname)
                print(f"Eliminando contenido de: {carpeta_aumentoDatos}")
                for filename in os.listdir(carpeta_aumentoDatos):
                    file_path = os.path.join(carpeta_aumentoDatos, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Error al eliminar {file_path}: {e}")

# Ruta raíz desde donde se iniciará la búsqueda
root_dir = '/home/ingsistemas/proyectobraille/ProyectoDeGrado/proyecto-braille-final'

eliminar_contenido_aumentoDatos(root_dir)