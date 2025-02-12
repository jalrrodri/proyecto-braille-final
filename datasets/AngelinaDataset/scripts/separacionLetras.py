import os
import csv
from PIL import Image

root = 'datasets/AngelinaDataset/books/chudo_derevo_redmi'
# --- Configuration ---
input_csv_dir = root + '/traducido'  # Directory containing the original CSV files
output_csv_dir = root + '/traducido/separado/anotaciones'  # Directory to save the updated CSV files
output_images_dir = root + '/traducido/separado'  # Directory where cropped images will be saved

# Create the output directories if they don't exist
os.makedirs(output_csv_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)

# Process each CSV file in the input directory
for csv_filename in os.listdir(input_csv_dir):
    if csv_filename.lower().endswith('.csv'):
        input_csv_path = os.path.join(input_csv_dir, csv_filename)
        
        # Add '_separados' to the output CSV filename
        name, ext = os.path.splitext(csv_filename)
        output_csv_filename = f"{name}_separados{ext}"
        output_csv_path = os.path.join(output_csv_dir, output_csv_filename)

        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csv_in, \
             open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_out:
            
            reader = csv.reader(csv_in)
            writer = csv.writer(csv_out)
            
            # If your CSV file has a header, uncomment the following two lines:
            # header = next(reader)
            # writer.writerow(header)
            
            for row in reader:
                # Skip empty rows
                if not row:
                    continue
                
                # Expected row structure:
                # [image_path, label, left, top, right, bottom]
                orig_image_path = row[0]
                label = row[1]
                
                try:
                    # Convert normalized coordinates (assumed in [0, 1)) to floats.
                    left_norm   = float(row[2])
                    top_norm    = float(row[3])
                    right_norm  = float(row[4])
                    bottom_norm = float(row[5])
                except ValueError as ve:
                    print(f"Error parsing coordinates in row {row}: {ve}")
                    continue
                
                # Open the original image
                try:
                    image = Image.open(orig_image_path)
                except Exception as e:
                    print(f"Error opening image {orig_image_path}: {e}")
                    continue

                # Get image dimensions
                img_width, img_height = image.size
                
                # Convert normalized coordinates to pixel coordinates.
                # Ensure that these values are within the image bounds.
                left_px   = int(left_norm * img_width)
                top_px    = int(top_norm * img_height)
                right_px  = int(right_norm * img_width)
                bottom_px = int(bottom_norm * img_height)
                
                # Crop the image. The box is (left, top, right, bottom)
                cropped_image = image.crop((left_px, top_px, right_px, bottom_px))
                
                # Generate a new filename for the cropped image.
                # This example uses the original image's basename, appending the label and pixel coordinates.
                base_name = os.path.basename(orig_image_path)
                name, ext = os.path.splitext(base_name)
                new_filename = f"{name}_{label}_{left_px}_{top_px}_{right_px}_{bottom_px}{ext}"
                new_image_path = os.path.join(output_images_dir, new_filename)
                
                # Save the cropped image.
                try:
                    cropped_image.save(new_image_path)
                except Exception as e:
                    print(f"Error saving cropped image {new_image_path}: {e}")
                    continue
                
                # Update the CSV row: replace the original image path with the new cropped image path.
                row[0] = new_image_path.replace("\\", "/")
                
                # Replace the last 4 columns with the specified 8 columns
                row = row[:2] + ['0.0', '0.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0']
                
                # Write the updated row to the output CSV.
                writer.writerow(row)

print("Processing complete. Cropped images saved and CSVs updated.")
