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
ruta_modelo = Path('modeloGuardado/optimizado/test')
ruta_modelo.mkdir(parents=True, exist_ok=True)

# Verificar versión de TensorFlow
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Nombre modelo
nombre_modelo = 'efficientdet_lite0'

# Especificación del modelo
spec = model_spec.get('efficientdet_lite0')

# Cargar datos desde CSV
try:
    train_data, validation_data, test_data = object_detector.DataLoader.from_csv('anotacionesIgualadasPrueba.csv')
except Exception as e:
    print(f"Error al cargar datos CSV: {e}")
    exit(1)

# Crear y entrenar el modelo
model = object_detector.create(
    train_data, 
    model_spec=spec, 
    epochs=1,  # Aumentar epochs para mejor rendimiento
    batch_size=8, 
    train_whole_model=False, 
    validation_data=validation_data
)

# Evaluar modelo
evaluation_results = model.evaluate(test_data)
print("Resultados de la Evaluación:", evaluation_results)

# Exportar modelo a TFLite
model.export(export_dir=str(ruta_modelo))

# Verificar si el archivo TFLite se ha creado correctamente
tflite_model_path = ruta_modelo / "model.tflite"
print(f"Ruta esperada del modelo TFLite: {tflite_model_path}")
if not tflite_model_path.exists():
    print(f"Error: El archivo TFLite no se ha creado en la ruta: {tflite_model_path}")
    exit(1)

# Evaluar el modelo TFLite
tflite_evaluation_results = model.evaluate_tflite(str(tflite_model_path), test_data)
print("Resultados de la Evaluación TFLite:", tflite_evaluation_results)

# Imprimir métricas de evaluación
print("\nMétricas de Evaluación del Modelo:")
for key, value in evaluation_results.items():
    print(f"{key}: {value}")

print("\nMétricas de Evaluación del Modelo TFLite:")
for key, value in tflite_evaluation_results.items():
    print(f"{key}: {value}")

# Leyenda de las métricas
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

# Ordenar las claves de las letras en los resultados de evaluación
evaluation_results = dict(sorted(evaluation_results.items()))
tflite_evaluation_results = dict(sorted(tflite_evaluation_results.items()))

# Graficar resultados y guardarlos en una ruta
ruta_graficos = Path('graficos')
ruta_graficos.mkdir(parents=True, exist_ok=True)

# Crear figura y ejes para las gráficas
fig, axs = plt.subplots(2, 1, figsize=(20, 15))

# Graficar métricas de evaluación del modelo
axs[0].bar(evaluation_results.keys(), evaluation_results.values(), color='blue', label="Modelo")
axs[0].set_ylabel('Valor')
axs[0].set_title('Métricas de Evaluación del Modelo')
axs[0].tick_params(axis='x', rotation=90)

# Graficar métricas de evaluación del modelo TFLite
axs[1].bar(tflite_evaluation_results.keys(), tflite_evaluation_results.values(), color='green', label="Modelo TFLite")
axs[1].set_ylabel('Valor')
axs[1].set_title('Métricas de Evaluación del Modelo TFLite')
axs[1].tick_params(axis='x', rotation=90)

# Añadir título principal al gráfico
fig.suptitle(nombre_modelo, fontsize=16)

# Añadir leyenda como un cuadro flotante a la derecha
legend_text = "\n".join([f"{clave}: {significado}" for clave, significado in leyenda.items()])
fig.legend([legend_text], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=True)

plt.tight_layout()
plt.savefig(ruta_graficos / 'resultados_evaluacion_' + nombre_modelo + '_prueba.png', bbox_inches='tight')
plt.close()