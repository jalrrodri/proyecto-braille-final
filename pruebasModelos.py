import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tflite_model_maker import object_detector
from tflite_support import metadata
from tensorflow.lite.python.interpreter import Interpreter
import cv2
from pathlib import Path
import json

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
MODEL_NAMES = [
    "efficientdet_lite0",
    "efficientdet_lite1",
    "efficientdet_lite2",
    "efficientdet_lite3"
]

# Colors for visualization (one for each letter A-Z)
COLORS = {}
for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    COLORS[letter] = tuple(np.random.randint(0, 255, 3).tolist())


def load_models(base_path="modelosGuardados"):
    """Carga los modelos TFLite y los retorna en tipo diccionario."""
    models = {}
    for model_name in MODEL_NAMES:
        model_path = Path(base_path) / model_name / "/optimizado/main/model.tflite"
        if not model_path.exists():
            print(f"Advertencia: Modelo {model_name} no encontrado en: {model_path}")
            continue

        print(f"Loading model: {model_name}")
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Get label map from metadata
        try:
            displayer = metadata.MetadataDisplayer.with_model_file(str(model_path))
            metadata_json = displayer.get_metadata_json()
            metadata_dict = json.loads(metadata_json)

            # Extract label map
            label_map = {}
            for i, label in enumerate(metadata_dict["associatedFiles"][0]["labelMap"]):
                label_map[i] = label
        except Exception as e:
            print(f"Error loading metadata for {model_name}: {e}")
            # Fallback to default A-Z label map
            label_map = {i: chr(65 + i) for i in range(26)}  # A-Z

        models[model_name] = {
            "interpreter": interpreter,
            "input_details": input_details,
            "output_details": output_details,
            "label_map": label_map,
        }

    return models


def load_test_images(csv_path="imagenesPrueba.csv", num_samples=26):
    """Load test images with ground truth annotations."""
    try:
        _, _, test_data = object_detector.DataLoader.from_csv(csv_path)

        # Get a subset of test data
        test_samples = []
        for i in range(min(num_samples, len(test_data))):
            test_samples.append(test_data[i])

        return test_samples
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []


def preprocess_image(image_path, input_shape):
    """Preprocess image for model input."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    original_image = image
    image = tf.image.resize(image, input_shape)
    return image, original_image


def run_inference(model_info, image):
    """Run inference on a single image."""
    interpreter = model_info["interpreter"]
    input_details = model_info["input_details"]
    output_details = model_info["output_details"]

    # Get input shape
    input_shape = input_details[0]["shape"][1:3]  # Height, width

    # Preprocess image
    processed_image, original_image = preprocess_image(image, input_shape)

    # Add batch dimension
    input_data = np.expand_dims(processed_image, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensors
    # For object detection, usually there are 4 output tensors:
    # - Locations (bounding boxes)
    # - Classes (class ids)
    # - Scores (confidence)
    # - Number of detections

    # Extract detection results
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]  # Bounding boxes
    classes = interpreter.get_tensor(output_details[1]["index"])[0]  # Class IDs
    scores = interpreter.get_tensor(output_details[2]["index"])[0]  # Confidence scores
    num_detections = int(
        interpreter.get_tensor(output_details[3]["index"])[0]
    )  # Number of detections

    return {
        "boxes": boxes[:num_detections],
        "classes": classes[:num_detections],
        "scores": scores[:num_detections],
        "num_detections": num_detections,
    }


def visualize_predictions(
    image_path, ground_truth, predictions_by_model, output_dir="comparison_results"
):
    """Visualize and save comparison between different models."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width, _ = image.shape

    # Create figure with subplots
    n_models = len(predictions_by_model) + 1  # +1 for ground truth
    fig, axs = plt.subplots(1, n_models, figsize=(n_models * 5, 5))

    # Plot ground truth
    axs[0].imshow(image)
    axs[0].set_title("Ground Truth")

    # Draw ground truth boxes
    for annotation in ground_truth:
        xmin, ymin, xmax, ymax = annotation["bbox"]
        xmin, ymin, xmax, ymax = (
            int(xmin * width),
            int(ymin * height),
            int(xmax * width),
            int(ymax * height),
        )
        label = annotation["label"]
        color = COLORS.get(label, (255, 0, 0))

        # Draw rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        # Add label
        label_text = f"{label}"
        cv2.putText(
            image,
            label_text,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    axs[0].imshow(image)
    axs[0].axis("off")

    # Plot predictions for each model
    for i, (model_name, predictions) in enumerate(predictions_by_model.items(), 1):
        # Make a copy of the original image
        pred_image = image.copy()

        # Get model info
        model_info = models[model_name]
        label_map = model_info["label_map"]

        # Draw prediction boxes
        for j in range(predictions["num_detections"]):
            if predictions["scores"][j] < 0.5:  # Skip low confidence detections
                continue

            # Get bounding box coordinates
            ymin, xmin, ymax, xmax = predictions["boxes"][j]
            xmin, ymin, xmax, ymax = (
                int(xmin * width),
                int(ymin * height),
                int(xmax * width),
                int(ymax * height),
            )

            # Get class ID and label
            class_id = int(predictions["classes"][j])
            label = label_map.get(class_id, f"Unknown ({class_id})")

            # Get color for this label
            color = COLORS.get(label, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(pred_image, (xmin, ymin), (xmax, ymax), color, 2)

            # Add label with confidence score
            confidence = predictions["scores"][j]
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(
                pred_image,
                label_text,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Show image with predictions
        axs[i].imshow(pred_image)
        axs[i].set_title(f"Model: {model_name}")
        axs[i].axis("off")

    # Save the figure
    image_name = os.path.basename(image_path)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"comparison_{image_name}"), bbox_inches="tight"
    )
    plt.close()

    print(f"Saved comparison for {image_path}")


def main():
    # Load all models
    global models
    models = load_models()

    if not models:
        print("No models loaded. Exiting.")
        return

    # Load test images
    test_samples = load_test_images(num_samples=26)

    if not test_samples:
        print("No test samples loaded. Exiting.")
        return

    # Create output directory
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)

    # Process each test sample
    for i, sample in enumerate(test_samples):
        print(f"Processing sample {i + 1}/{len(test_samples)}")

        # Get image path and ground truth
        image_path = sample["image"]
        ground_truth = sample["objects"]

        # Run inference with each model
        predictions_by_model = {}
        for model_name, model_info in models.items():
            try:
                predictions = run_inference(model_info, image_path)
                predictions_by_model[model_name] = predictions
            except Exception as e:
                print(f"Error running inference with {model_name}: {e}")

        # Visualize predictions
        visualize_predictions(
            image_path, ground_truth, predictions_by_model, output_dir
        )


if __name__ == "__main__":
    main()
