import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# Define input and output paths here
INPUT_DIR = "datasets/libroCorazonDelator/completas/anotaciones/xml"  # Directory containing XML files
OUTPUT_FILE = "datasets/libroCorazonDelator/completas/anotaciones/csv/anotaciones.csv"  # Output CSV file


def xml_to_csv(input_dir, output_file):
    """
    Convert Pascal VOC XML files to a single CSV file for TFLite Model Maker
    with normalized coordinates (0-1 range).
    """
    results = []

    # Find all XML files in the input directory
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image size for normalization
        width = float(root.find("size/width").text)
        height = float(root.find("size/height").text)

        # Get path to the image
        path = root.find("path").text if root.find("path") is not None else ""
        filename = root.find("filename").text

        # If path is not provided, construct from folder and filename
        if not path or os.path.basename(path) != filename:
            folder = root.find("folder").text if root.find("folder") is not None else ""
            path = os.path.join("datasets", folder, filename)
        else:
            # Extract path relative to dataset directory
            parts = path.split("datasets")
            if len(parts) > 1:
                path = "datasets" + parts[1]
            else:
                # If 'datasets' is not in the path, use the filename with folder
                folder = (
                    root.find("folder").text if root.find("folder") is not None else ""
                )
                path = os.path.join("datasets", folder, filename)

        # Process each object in the XML
        for member in root.findall("object"):
            name = member.find("name").text

            # Get bounding box coordinates
            xmin = float(member.find("bndbox/xmin").text)
            ymin = float(member.find("bndbox/ymin").text)
            xmax = float(member.find("bndbox/xmax").text)
            ymax = float(member.find("bndbox/ymax").text)

            # Normalize coordinates to 0-1 range
            xmin_norm = xmin / width
            ymin_norm = ymin / height
            xmax_norm = xmax / width
            ymax_norm = ymax / height

            # Format for TFLite Model Maker (keeping both raw and normalized coordinates)
            row = [
                path,  # Image path
                name,  # Class name
                xmin_norm,  # Raw xmin
                ymin_norm,  # Raw ymin
                0.0,  # Normalized xmin
                0.0,  # Normalized ymin
                xmax_norm,  # Raw xmax
                ymax_norm,  # Raw ymax
                0.0,  # Normalized xmax
                0.0,  # Normalized ymax
            ]
            results.append(row)

    # Create DataFrame and save to CSV
    if results:
        column_names = [
            "filename",
            "class",
            "xmin",
            "ymin",
            "xmin_norm",
            "ymin_norm",
            "xmax",
            "ymax",
            "xmax_norm",
            "ymax_norm",
        ]
        df = pd.DataFrame(results, columns=column_names)
        df.to_csv(output_file, index=False, header=False)
        print(f"Converted {len(xml_files)} XML files to {output_file}")
        return True
    else:
        print("No XML files found or processed.")
        return False


def main():
    # Using hardcoded paths instead of command line arguments
    xml_to_csv(INPUT_DIR, OUTPUT_FILE)
    print(f"Processing complete. Check {OUTPUT_FILE} for results.")


if __name__ == "__main__":
    main()
