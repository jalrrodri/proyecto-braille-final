# requiere:
# python 3.6.8 https://www.python.org/downloads/release/python-368/
#     pip install tflite-model-maker
#     pip install tensorflow
#     pip install cmake
#     pip install pycocotools
# visual studio con desarrollo para el escritorio con C++

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

# Importa Path para trabajar con rutas de archivos
from pathlib import Path

# Importa matplotlib
import matplotlib.pyplot as plt

ruta_modelo = Path('modeloGuardado/optimizado/efficientdet_lite0')

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite0')
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('anotaciones.csv')
model = object_detector.create(train_data, model_spec=spec, epochs=1, batch_size=8, train_whole_model=True, validation_data=validation_data)
model.evaluate(test_data)
evaluation_results = model.export(export_dir=ruta_modelo)

# Evaluar el modelo TFLite
tflite_evaluation_results = model.evaluate_tflite("modeloGuardado/optimizado/mobilenet/model.tflite", test_data)

# Crear gráficos para las métricas de evaluación
plt.figure(figsize=(10, 5))

# Precisión del modelo en conjunto de datos de prueba
plt.subplot(1, 2, 1)
plt.bar(["Modelo", "TFLite"], [evaluation_results["accuracy"], tflite_evaluation_results["accuracy"]], color=['blue', 'green'])
plt.ylabel('Precisión')
plt.title('Precisión del Modelo y TFLite')

# Pérdida del modelo en conjunto de datos de prueba
plt.subplot(1, 2, 2)
plt.bar(["Modelo", "TFLite"], [evaluation_results["loss"], tflite_evaluation_results["loss"]], color=['blue', 'green'])
plt.ylabel('Pérdida')
plt.title('Pérdida del Modelo y TFLite')

plt.tight_layout()
plt.show()

