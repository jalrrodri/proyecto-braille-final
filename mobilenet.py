from tflite_model_maker import model_spec, object_detector
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

# Establecer semillas aleatorias para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Configuración de rutas
ruta_modelo = Path('modeloGuardado/optimizado/efficientdet_lite0')
ruta_modelo.mkdir(parents=True, exist_ok=True)

# Verificar versión de TensorFlow
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Especificación del modelo
spec = model_spec.get('efficientdet_lite0')

# Cargar datos desde CSV
try:
    train_data, validation_data, test_data = object_detector.DataLoader.from_csv('anotaciones.csv')
except Exception as e:
    print(f"Error al cargar datos CSV: {e}")
    exit(1)

# Crear y entrenar el modelo
model = object_detector.create(
    train_data, 
    model_spec=spec, 
    epochs=1,  # Aumentar epochs para mejor rendimiento
    batch_size=8, 
    train_whole_model=True, 
    validation_data=validation_data
)

# Evaluar modelo
evaluation_results = model.evaluate(test_data)
print("Resultados de la Evaluación:", evaluation_results)

# Exportar modelo a TFLite
model.export(export_dir=str(ruta_modelo))

# Evaluar el modelo TFLite
tflite_model_path = ruta_modelo / "model.tflite"
tflite_evaluation_results = model.evaluate_tflite(str(tflite_model_path), test_data)
print("Resultados de la Evaluación TFLite:", tflite_evaluation_results)

# Graficar resultados y guardarlos en una ruta
ruta_graficos = Path('graficos')
ruta_graficos.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))

# Precisión del modelo en conjunto de datos de prueba
plt.subplot(1, 2, 1)
plt.bar(
    ["Modelo", "TFLite"], 
    [evaluation_results.get("AP", 0), tflite_evaluation_results.get("AP", 0)], 
    color=['blue', 'green']
)
plt.ylabel('Precisión Promedio (AP)')
plt.title('Precisión del Modelo y TFLite')

# Pérdida del modelo en conjunto de datos de prueba
plt.subplot(1, 2, 2)
plt.bar(
    ["Modelo", "TFLite"], 
    [evaluation_results.get("loss", 0), tflite_evaluation_results.get("loss", 0)], 
    color=['blue', 'green']
)
plt.ylabel('Pérdida')
plt.title('Pérdida del Modelo y TFLite')

plt.tight_layout()
plt.savefig(ruta_graficos / 'resultados_evaluacion.png')
plt.close()
