from tflite_model_maker import model_spec, object_detector
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from absl import logging
import os
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Establecer semillas aleatorias para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Verificar versión de TensorFlow
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

# Lista de modelos a entrenar
OBJECT_DETECTION_MODELS = [
    'efficientdet_lite0',
    'efficientdet_lite1',
    'efficientdet_lite2',
    'efficientdet_lite3',
    'efficientdet_lite4',
]

# Configuración de rutas base
ruta_base = Path('modelosGuardados')
ruta_graficos = Path('graficos')
ruta_graficos.mkdir(parents=True, exist_ok=True)

# Cargar datos desde CSV (solo una vez para todos los modelos)
print("Cargando datos desde CSV...")
try:
    train_data, validation_data, test_data = object_detector.DataLoader.from_csv('anotacionesIgualadas.csv')
    print(f"Datos cargados correctamente: {len(train_data)} entrenamiento, {len(validation_data)} validación, {len(test_data)} prueba")
except Exception as e:
    print(f"Error al cargar datos CSV: {e}")
    exit(1)

# Leyenda de las métricas para la gráfica final
leyenda = {
    "AP": "Precisión promedio en diferentes umbrales de IoU",
    "AP50": "Precisión promedio con IoU ≥ 50%",
    "AP75": "Precisión promedio con IoU ≥ 75%",
    "APs": "Precisión en objetos pequeños",
    "APm": "Precisión en objetos medianos",
    "APl": "Precisión en objetos grandes",
    "ARmax1": "Recall máximo con 1 detección por imagen",
    "ARmax10": "Recall máximo con 10 detecciones por imagen",
    "ARmax100": "Recall máximo con 100 detecciones por imagen",
    "ARs": "Recall en objetos pequeños",
    "ARm": "Recall en objetos medianos",
    "ARl": "Recall en objetos grandes",
}

# Agregar las categorías individuales (A-Z) en orden alfabético
for letra in sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    leyenda[f"AP_/{letra}"] = f"Precisión promedio para la letra {letra}"

# Resultados para comparación final
todos_los_resultados = {}

# Iterar sobre cada modelo
for nombre_modelo in OBJECT_DETECTION_MODELS:
    print(f"\n{'='*80}")
    print(f"Iniciando entrenamiento del modelo: {nombre_modelo}")
    print(f"{'='*80}")
    
    # Configuración de rutas para este modelo
    ruta_modelo = ruta_base / nombre_modelo / 'optimizado/main'
    ruta_modelo.mkdir(parents=True, exist_ok=True)
    
    try:
        # Obtener especificación del modelo usando directamente el string
        print(f"Obteniendo especificación del modelo {nombre_modelo}...")
        spec = model_spec.get(nombre_modelo)
        
        # Crear y entrenar el modelo
        print(f"Creando y entrenando el modelo {nombre_modelo}...")
        inicio_entrenamiento = time.time()
        model = object_detector.create(
            train_data=train_data, 
            model_spec=spec, 
            epochs=50,  # Aumentar epochs para mejor rendimiento
            batch_size=16, 
            train_whole_model=True, 
            validation_data=validation_data,
            do_train=True
        )
        tiempo_entrenamiento = time.time() - inicio_entrenamiento
        print(f"Entrenamiento completado en {tiempo_entrenamiento:.2f} segundos")
        
        # Evaluar modelo
        print(f"Evaluando modelo {nombre_modelo}...")
        evaluation_results = model.evaluate(test_data)
        print("Resultados de la Evaluación:", evaluation_results)
        
        # Exportar modelo a TFLite
        print(f"Exportando modelo {nombre_modelo} a TFLite...")
        model.export(export_dir=str(ruta_modelo))
        
        # Verificar si el archivo TFLite se ha creado correctamente
        tflite_model_path = ruta_modelo / "model.tflite"
        print(f"Ruta esperada del modelo TFLite: {tflite_model_path}")
        if not tflite_model_path.exists():
            print(f"Error: El archivo TFLite no se ha creado en la ruta: {tflite_model_path}")
            continue
        
        # Evaluar el modelo TFLite
        print(f"Evaluando modelo TFLite {nombre_modelo}...")
        tflite_evaluation_results = model.evaluate_tflite(str(tflite_model_path), test_data)
        print("Resultados de la Evaluación TFLite:", tflite_evaluation_results)
        
        # Guardar los resultados para comparación final
        todos_los_resultados[nombre_modelo] = {
            'model': dict(sorted(evaluation_results.items())),
            'tflite': dict(sorted(tflite_evaluation_results.items())),
            'training_time': tiempo_entrenamiento
        }
        
        # Imprimir métricas de evaluación
        print(f"\nMétricas de Evaluación del Modelo {nombre_modelo}:")
        for key, value in sorted(evaluation_results.items()):
            print(f"{key}: {value}")
        
        print(f"\nMétricas de Evaluación del Modelo TFLite {nombre_modelo}:")
        for key, value in sorted(tflite_evaluation_results.items()):
            print(f"{key}: {value}")
        
        # Crear figura y ejes para las gráficas específicas de este modelo
        fig, axs = plt.subplots(2, 1, figsize=(20, 15))
        
        # Graficar métricas de evaluación del modelo
        axs[0].bar(evaluation_results.keys(), evaluation_results.values(), color='blue', label="Modelo")
        axs[0].set_ylabel('Valor')
        axs[0].set_title(f'Métricas de Evaluación del Modelo {nombre_modelo}')
        axs[0].tick_params(axis='x', rotation=90)
        
        # Graficar métricas de evaluación del modelo TFLite
        axs[1].bar(tflite_evaluation_results.keys(), tflite_evaluation_results.values(), color='green', label="Modelo TFLite")
        axs[1].set_ylabel('Valor')
        axs[1].set_title(f'Métricas de Evaluación del Modelo TFLite {nombre_modelo}')
        axs[1].tick_params(axis='x', rotation=90)
        
        # Añadir título principal al gráfico
        fig.suptitle(nombre_modelo, fontsize=16)
        
        # Añadir leyenda como un cuadro flotante a la derecha
        legend_text = "\n".join([f"{clave}: {significado}" for clave, significado in leyenda.items()])
        fig.legend([legend_text], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=True)
        
        plt.tight_layout()
        plt.savefig(ruta_graficos / f'resultados_evaluacion_{nombre_modelo}.png', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error durante el procesamiento del modelo {nombre_modelo}: {e}")
        continue

# Crear gráficas de comparación de todos los modelos
if todos_los_resultados:
    print("\nCreando gráficas comparativas de todos los modelos...")
    
    # Comparar métricas comunes entre modelos
    common_metrics = ['AP', 'AP50', 'AP75']
    
    # Crear figura para comparación de modelos
    fig, axs = plt.subplots(3, 1, figsize=(15, 18))
    
    # Comparar AP, AP50, AP75 entre modelos
    for i, metric in enumerate(common_metrics):
        model_values = [todos_los_resultados[model]['model'].get(metric, 0) for model in OBJECT_DETECTION_MODELS if model in todos_los_resultados]
        tflite_values = [todos_los_resultados[model]['tflite'].get(metric, 0) for model in OBJECT_DETECTION_MODELS if model in todos_los_resultados]
        models = [model for model in OBJECT_DETECTION_MODELS if model in todos_los_resultados]
        
        x = range(len(models))
        width = 0.35
        
        axs[i].bar([p - width/2 for p in x], model_values, width, label='Modelo Original', color='blue')
        axs[i].bar([p + width/2 for p in x], tflite_values, width, label='Modelo TFLite', color='green')
        
        axs[i].set_ylabel(f'{metric} Value')
        axs[i].set_title(f'Comparación de {metric} entre modelos')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(models)
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(ruta_graficos / 'comparacion_modelos_metricas.png', bbox_inches='tight')
    plt.close()
    
    # Comparar tiempos de entrenamiento
    plt.figure(figsize=(10, 6))
    training_times = [todos_los_resultados[model]['training_time'] / 60 for model in OBJECT_DETECTION_MODELS if model in todos_los_resultados]  # Convertir a minutos
    models = [model for model in OBJECT_DETECTION_MODELS if model in todos_los_resultados]
    
    plt.bar(models, training_times, color='purple')
    plt.ylabel('Tiempo de entrenamiento (minutos)')
    plt.title('Comparación de tiempos de entrenamiento')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ruta_graficos / 'comparacion_tiempos_entrenamiento.png')
    plt.close()
    
    print("\nProceso completo. Resultados guardados en el directorio 'graficos'")
else:
    print("\nNo se pudieron procesar correctamente ninguno de los modelos")