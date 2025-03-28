import os
import shutil
from pathlib import Path

def clean_resize_folders(root_dir: str) -> None:
    """
    Busca y limpia todas las carpetas llamadas 'redimensionamiento' 
    bajo el directorio raíz especificado.
    
    Args:
        root_dir (str): Directorio raíz donde comenzar la búsqueda
    """
    root_path = Path(root_dir)
    
    # Buscar todas las carpetas llamadas 'redimensionamiento'
    resize_folders = root_path.rglob('redimensionado')
    
    for folder in resize_folders:
        if folder.is_dir():
            try:
                # Eliminar todo el contenido de la carpeta
                shutil.rmtree(folder)
                # Recrear la carpeta vacía
                folder.mkdir(exist_ok=True)
                print(f"✓ Limpiada carpeta: {folder}")
            except Exception as e:
                print(f"✗ Error al limpiar {folder}: {str(e)}")

if __name__ == "__main__":
    # Directorio raíz del proyecto
    PROJECT_ROOT = Path(__file__).parent
    
    print("Iniciando limpieza de carpetas 'redimensionamiento'...")
    clean_resize_folders(PROJECT_ROOT)
    print("Proceso completado.")