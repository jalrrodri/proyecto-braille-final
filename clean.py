import os
import shutil
import stat

def eliminar_contenido_traducido(root_dir):
    # Recorrer el sistema de archivos desde la raíz especificada
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Verificar si la carpeta actual se llama 'traducido', 'datasetprueba1FILTROS' o 'datasetprueba1AUMENTADO'
        if os.path.basename(dirpath) in ['traducido', 'datasetprueba1FILTROS', 'datasetprueba1AUMENTADO']:
            print(f"Eliminando contenido de la carpeta: {dirpath}")
            # Eliminar todos los archivos en la carpeta
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    # Quitar el atributo de solo lectura si está presente
                    os.chmod(file_path, stat.S_IWRITE)
                    os.remove(file_path)
                    print(f"Archivo eliminado: {file_path}")
                except PermissionError:
                    print(f"Permiso denegado al eliminar el archivo {file_path}")
                except Exception as e:
                    print(f"Error al eliminar el archivo {file_path}: {e}")
            # Eliminar todas las subcarpetas en la carpeta
            for dirname in dirnames:
                dir_to_remove = os.path.join(dirpath, dirname)
                try:
                    # Quitar el atributo de solo lectura si está presente
                    os.chmod(dir_to_remove, stat.S_IWRITE)
                    shutil.rmtree(dir_to_remove)
                    print(f"Carpeta eliminada: {dir_to_remove}")
                except PermissionError:
                    print(f"Permiso denegado al eliminar la carpeta {dir_to_remove}")
                except Exception as e:
                    print(f"Error al eliminar la carpeta {dir_to_remove}: {e}")

# Especificar la raíz desde donde se comenzará a buscar
root_dir = 'd:/ProyectoDeGrado/proyecto-braille-final'

eliminar_contenido_traducido(root_dir)