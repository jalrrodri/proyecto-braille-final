import csv
import os
import random

def generate_csv(folder_path, output_filename):
  """
  Generates a CSV file from a folder containing JPEGs, assigning random labels.

  Args:
    folder_path: Path to the folder containing JPEGs.
    output_filename: Name of the output CSV file.
  """
  data = []
  train_ratio = 0.85
  val_ratio = 0.10
  test_ratio = 0.05

  for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg"):
      filepath = os.path.join(folder_path, filename)
      first_letter = filename[0].upper()

      # Randomly assign label based on ratios
      label_options = ["TRAIN"] * int(train_ratio * 100) + ["VALIDATION"] * int(val_ratio * 100) + ["TEST"] * int(test_ratio * 100)
      label = random.choice(label_options)

      # Modify the second column to include "gs://datasetbraille/"
      cloud_storage_path = f"gs://datasetbraille/{filename}"

      data.append([label, cloud_storage_path, f"Letra: {first_letter}", 0,0,1,0,1,1,0,1])

  with open(output_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)

  print(f"CSV file generated successfully: {output_filename}")

# Replace these with your actual folder path and desired output filename
folder_path = "dataset"
output_filename = "dataset2.csv"

generate_csv(folder_path, output_filename)
