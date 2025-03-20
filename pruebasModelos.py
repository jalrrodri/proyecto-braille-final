import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
import os
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Configuration
MODEL_NAMES = [
    "efficientdet_lite0",
    "efficientdet_lite1",
    "efficientdet_lite2",
    "efficientdet_lite3",
]
MODEL_BASE_PATH = Path("modelosGuardados")
IMAGE_PATH = Path("imagenesPrueba")  # Directory with test images
OUTPUT_PATH = Path("graficosPruebas")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# Load ground truth labels (assuming you have a CSV with image names and real labels)
# Modify this according to your data format
def load_ground_truth(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Use the third column for the label and convert to uppercase
        label_column = df.columns[2]  # Get the name of the third column
        return {row["image_name"]: row[label_column].upper() for _, row in df.iterrows()}
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        # If CSV not available, extract first letter and convert to uppercase
        image_files = list(IMAGE_PATH.glob("*.jpg")) + list(IMAGE_PATH.glob("*.png"))
        # Extract only the first letter from the filename and convert to uppercase
        return {img.name: img.name[0].upper() for img in image_files}


# Load TFLite models
def load_model(model_name):
    model_path = Path(MODEL_BASE_PATH).joinpath(
        model_name, "optimizado", "main", "model.tflite"
    )
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None

    # Create TFLite task detector
    base_options = core.BaseOptions(file_name=str(model_path))
    detection_options = processor.DetectionOptions(max_results=5, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options
    )

    try:
        detector = vision.ObjectDetector.create_from_options(options)
        return detector
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None


# Process an image with the model
def process_image(detector, image_path):
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None, []

    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create TensorImage from numpy array
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run inference
    detection_result = detector.detect(input_tensor)

    # Get top detection
    if detection_result.detections:
        # Sort by score
        sorted_detections = sorted(
            detection_result.detections,
            key=lambda x: x.categories[0].score,
            reverse=True,
        )

        top_detection = sorted_detections[0]
        label = top_detection.categories[0].category_name
        confidence = top_detection.categories[0].score

        # Draw bounding box on image
        bbox = top_detection.bounding_box
        cv2.rectangle(
            image,
            (bbox.origin_x, bbox.origin_y),
            (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
            (0, 255, 0),
            2,
        )

        # Add label and confidence
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


# Main function
def main():
    # Load ground truth
    ground_truth = load_ground_truth("imagenesPrueba.csv")  # Update with your CSV path

    # Get test images
    image_files = list(IMAGE_PATH.glob("*.jpg")) + list(IMAGE_PATH.glob("*.png"))
    if len(image_files) == 0:
        print(f"No images found in {IMAGE_PATH}")
        return

    # Use the first 26 images or all if fewer
    test_images = image_files[:26] if len(image_files) > 26 else image_files
    print(f"Testing with {len(test_images)} images")

    # Load models
    models = {}
    for model_name in MODEL_NAMES:
        detector = load_model(model_name)
        if detector:
            models[model_name] = detector

    if not models:
        print("No models loaded successfully")
        return

    # Process images with each model
    results = {model_name: [] for model_name in models}

    for image_path in test_images:
        image_name = image_path.name
        true_label = ground_truth.get(image_name, "Unknown")

        for model_name, detector in models.items():
            annotated_image, detections = process_image(detector, image_path)

            predicted_label = detections[0][0] if detections else "No detection"
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

    # Create visualization grid
    num_images = len(test_images)
    num_models = len(models)

    # Calculate grid dimensions
    rows = min(26, num_images)
    cols = num_models + 1  # +1 for original image

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    fig.suptitle("Object Detection Results Comparison", fontsize=16)

    # Set column titles
    if rows == 1:
        axes[0].set_title("Original (Ground Truth)", fontsize=12)
        for i, model_name in enumerate(models.keys()):
            axes[i + 1].set_title(model_name, fontsize=12)
    else:
        axes[0, 0].set_title("Original (Ground Truth)", fontsize=12)
        for i, model_name in enumerate(models.keys()):
            axes[0, i + 1].set_title(model_name, fontsize=12)

    # Plot results
    for row_idx in range(rows):
        image_path = test_images[row_idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        true_label = ground_truth.get(image_path.name, "Unknown")

        # Create a copy of the original image and add the ground truth label
        labeled_image = image.copy()
        # Add text with true label at the top of the image
        cv2.putText(
            labeled_image,
            f"True: {true_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Original image
        if rows == 1:
            ax = axes[0]
        else:
            ax = axes[row_idx, 0]

        ax.imshow(labeled_image)
        # We'll add the image name below the image
        if image_path.name:
            ax.set_xlabel(f"Image: {image_path.name}", fontsize=8)
        ax.axis("off")

        # Model results
        for col_idx, model_name in enumerate(models.keys()):
            result = results[model_name][row_idx]

            if rows == 1:
                ax = axes[col_idx + 1]
            else:
                ax = axes[row_idx, col_idx + 1]

            # Add ground truth to model output images as well
            result_img = result["annotated_image"].copy()
            # Add text with true label at the top of the image (in red to differentiate)
            cv2.putText(
                result_img,
                f"True: {true_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            ax.imshow(result_img)
            # Show prediction and confidence as title
            correct = result["predicted_label"] == true_label
            title_color = "green" if correct else "red"

            ax.set_title(
                f"Pred: {result['predicted_label']}\nConf: {result['confidence']:.2f}",
                fontsize=10,
                color=title_color,
            )
            ax.axis("off")

            # Add match/mismatch indicator
            match_text = "✓" if correct else "✗"
            ax.text(
                0.5,
                -0.1,
                match_text,
                fontsize=16,
                color=title_color,
                ha="center",
                transform=ax.transAxes,
            )

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)
    fig.savefig(OUTPUT_PATH / "detection_comparison.png", bbox_inches="tight", dpi=150)

    # Generate summary statistics
    print("\nSummary Statistics:")
    for model_name in models:
        correct = sum(
            1 for r in results[model_name] if r["predicted_label"] == r["true_label"]
        )
        accuracy = correct / len(results[model_name])
        print(
            f"{model_name}: Accuracy = {accuracy:.2f} ({correct}/{len(results[model_name])})"
        )

    # Generate confusion matrices
    for model_name in models:
        model_results = results[model_name]

        # Get unique labels (true and predicted)
        true_labels = set(r["true_label"] for r in model_results)
        pred_labels = set(r["predicted_label"] for r in model_results)
        all_labels = sorted(true_labels.union(pred_labels))

        # Create confusion matrix
        cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        for r in model_results:
            true_idx = label_to_idx.get(r["true_label"], -1)
            pred_idx = label_to_idx.get(r["predicted_label"], -1)
            if true_idx >= 0 and pred_idx >= 0:
                cm[true_idx, pred_idx] += 1

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.colorbar()

        tick_marks = np.arange(len(all_labels))
        plt.xticks(tick_marks, all_labels, rotation=45)
        plt.yticks(tick_marks, all_labels)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        # Add text annotations
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

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
