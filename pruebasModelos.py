import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tflite_model_maker import object_detector
from tflite_support import metadata
from tensorflow.lite.python.interpreter import Interpreter
import cv2
from pathlib import Path
import cv2
import pandas as pd
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Configuración
MODEL_NAMES = [
    "efficientdet_lite0",
    "efficientdet_lite1",
    "efficientdet_lite2",
    "efficientdet_lite3",
]
MODEL_BASE_PATH = Path("modelosGuardados")
IMAGE_PATH = Path("imagenesPrueba")  # Directorio con imágenes de prueba
OUTPUT_PATH = Path("graficosPruebas")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# Cargar etiquetas de verdad (asumiendo que tienes un CSV con nombres de imágenes y etiquetas reales)
# Modifica esto de acuerdo a tu formato de datos
def load_ground_truth(csv_path):
    try:
        df = pd.read_csv(csv_path, header=None)  # Tu CSV parece no tener encabezados
        
        # La segunda columna (índice 1) contiene las rutas de las imágenes
        # La tercera columna (índice 2) contiene las etiquetas
        
        # Extraer solo el nombre del archivo de la ruta completa
        # Por ejemplo, de "imagenesPrueba/a.jpg" obtener "a.jpg"
        image_dict = {}
        for _, row in df.iterrows():
            # Obtener la ruta completa y extraer solo el nombre del archivo
            image_path = row[1]  # Segunda columna (índice 1)
            image_name = Path(image_path).name  # Extrae solo "a.jpg" de "imagenesPrueba/a.jpg"
            label = row[2].upper()  # Tercera columna (índice 2), convertida a mayúsculas
            image_dict[image_name] = label
            
        return image_dict
    except Exception as e:
        print(f"Error al cargar las etiquetas de verdad: {e}")
        # Si el CSV no está disponible, usa el plan de respaldo
        image_files = list(IMAGE_PATH.glob("*.jpg")) + list(IMAGE_PATH.glob("*.png"))
        return {img.name: img.name[0].upper() for img in image_files}


# Cargar modelos TFLite
def load_model(model_name):
    model_path = Path(MODEL_BASE_PATH).joinpath(
        model_name, "optimizado", "main", "model.tflite"
    )
    if not model_path.exists():
        print(f"Modelo no encontrado en {model_path}")
        return None

    # Crear detector de tareas TFLite
    base_options = core.BaseOptions(file_name=str(model_path))
    detection_options = processor.DetectionOptions(max_results=5, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options
    )

    try:
        detector = vision.ObjectDetector.create_from_options(options)
        return detector
    except Exception as e:
        print(f"Error al cargar el modelo {model_name}: {e}")
        return None


# Procesar una imagen con el modelo
def process_image(detector, image_path):
    # Leer la imagen
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error al leer la imagen: {image_path}")
        return None, []

    # Convertir a RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crear TensorImage desde array de numpy
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Ejecutar inferencia
    detection_result = detector.detect(input_tensor)

    # Obtener la detección principal
    if detection_result.detections:
        # Ordenar por puntuación
        sorted_detections = sorted(
            detection_result.detections,
            key=lambda x: x.categories[0].score,
            reverse=True,
        )

        top_detection = sorted_detections[0]
        label = top_detection.categories[0].category_name
        confidence = top_detection.categories[0].score

        # Dibujar caja delimitadora en la imagen
        bbox = top_detection.bounding_box
        cv2.rectangle(
            image,
            (bbox.origin_x, bbox.origin_y),
            (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
            (0, 255, 0),
            2,
        )

        # Añadir etiqueta y confianza
        cv2.putText(
            image,
            f"{label}: {confidence:.2f}",
            (bbox.origin_x, bbox.origin_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        return image, [(label, confidence)]
    else:
        return image, []


# Función principal
def main():
    # Cargar etiquetas de verdad
    ground_truth = load_ground_truth("imagenesPrueba.csv")  # Actualiza con la ruta de tu CSV

    # Obtener imágenes de prueba
    image_files = list(IMAGE_PATH.glob("*.jpg")) + list(IMAGE_PATH.glob("*.png"))
    if len(image_files) == 0:
        print(f"No se encontraron imágenes en {IMAGE_PATH}")
        return

    # Usar las primeras 26 imágenes o todas si hay menos
    test_images = image_files[:26] if len(image_files) > 26 else image_files
    print(f"Probando con {len(test_images)} imágenes")

    # Cargar modelos
    models = {}
    for model_name in MODEL_NAMES:
        detector = load_model(model_name)
        if detector:
            models[model_name] = detector

    if not models:
        print("Ningún modelo se cargó correctamente")
        return

    # Procesar imágenes con cada modelo
    results = {model_name: [] for model_name in models}

    for image_path in test_images:
        image_name = image_path.name
        true_label = ground_truth.get(image_name, "Desconocido")

        for model_name, detector in models.items():
            annotated_image, detections = process_image(detector, image_path)

            predicted_label = detections[0][0] if detections else "Sin detección"
            confidence = detections[0][1] if detections else 0.0

            results[model_name].append(
                {
                    "image_path": image_path,
                    "image_name": image_name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "annotated_image": annotated_image,
                }
            )

    # Crear cuadrícula de visualización
    num_images = len(test_images)
    num_models = len(models)

    # Calcular dimensiones de la cuadrícula
    rows = min(26, num_images)
    cols = num_models + 1  # +1 para la imagen original

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle("Comparación de Resultados de Detección de Objetos", fontsize=16)

    # Establecer títulos de columnas
    if rows == 1:
        axes[0].set_title("Original (Verdad de Referencia)", fontsize=12)
        for i, model_name in enumerate(models.keys()):
            axes[i + 1].set_title(model_name, fontsize=12)
    else:
        axes[0, 0].set_title("Original (Verdad de Referencia)", fontsize=12)
        for i, model_name in enumerate(models.keys()):
            axes[0, i + 1].set_title(model_name, fontsize=12)

    # Visualizar resultados
    for row_idx in range(rows):
        # Visualizar resultados
        image_path = test_images[row_idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        true_label = ground_truth.get(image_path.name, "Desconocido")

        # Imagen original
        if rows == 1:
            ax = axes[0]
        else:
            ax = axes[row_idx, 0]

        ax.imshow(image)
        # Añadir etiqueta verdadera como título
        ax.set_title(f"Real: {true_label}", pad=10, color='red')
        # Añadir nombre de la imagen debajo
        if image_path.name:
            ax.set_xlabel(f"Imagen: {image_path.name}", fontsize=8)
        ax.axis("off")

        # Resultados del modelo
        for col_idx, model_name in enumerate(models.keys()):
            result = results[model_name][row_idx]

            if rows == 1:
                ax = axes[col_idx + 1]
            else:
                ax = axes[row_idx, col_idx + 1]

            # Añadir etiqueta verdadera a las imágenes de salida del modelo también
            result_img = result["annotated_image"].copy()
            ax.imshow(result_img)
            # Mostrar predicción y confianza como título
            correct = result["predicted_label"] == true_label
            title_color = "green" if correct else "red"

            ax.set_title(
                f"Real: {true_label} | Pred: {result['predicted_label']}\nConf: {result['confidence']:.2f}\n{'✓' if correct else '✗'}",
                fontsize=10,
                color=title_color,
                pad=10,
            )
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)
    fig.savefig(OUTPUT_PATH / "detection_comparison.png", bbox_inches="tight", dpi=150)

    # Generar estadísticas resumidas
    print("\nEstadísticas Resumidas:")
    for model_name in models:
        correct = sum(
            1 for r in results[model_name] if r["predicted_label"] == r["true_label"]
        )
        accuracy = correct / len(results[model_name])
        print(
            f"{model_name}: Precisión = {accuracy:.2f} ({correct}/{len(results[model_name])})"
        )

    # Generar matrices de confusión
    for model_name in models:
        model_results = results[model_name]

        # Obtener etiquetas únicas (verdaderas y predichas)
        true_labels = set(r["true_label"] for r in model_results)
        pred_labels = set(r["predicted_label"] for r in model_results)
        all_labels = sorted(true_labels.union(pred_labels))

        # Crear matriz de confusión
        cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        for r in model_results:
            true_idx = label_to_idx.get(r["true_label"], -1)
            pred_idx = label_to_idx.get(r["predicted_label"], -1)
            if true_idx >= 0 and pred_idx >= 0:
                cm[true_idx, pred_idx] += 1

        # Visualizar matriz de confusión
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión - {model_name}")
        plt.colorbar()

        tick_marks = np.arange(len(all_labels))
        plt.xticks(tick_marks, all_labels, rotation=45)
        plt.yticks(tick_marks, all_labels)

        plt.xlabel("Etiqueta Predicha")
        plt.ylabel("Etiqueta Real")
        plt.tight_layout()

        # Añadir anotaciones de texto
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.savefig(
            OUTPUT_PATH / f"confusion_matrix_{model_name}.png", bbox_inches="tight"
        )
        plt.close()

    print(f"\nResultados guardados en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()