import os
import csv
import shutil

def process_csv_file(input_path, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read and process the CSV file
    with open(input_path, 'r') as input_file:
        reader = csv.reader(input_file)
        rows = list(reader)
        
        # Process each row
        for row in rows:
            # Insert "0.0" at positions 3,4,7,8
            # Note: We insert from right to left to maintain correct positions
            row.insert(8, "0.0")
            row.insert(7, "0.0")
            row.insert(4, "0.0")
            row.insert(4, "0.0")
    
    # Write the modified data to the output file
    with open(output_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(rows)

def main():
    root_paths = [
        "datasets/AngelinaDataset/books/chudo_derevo_redmi/traducido"
    ]
    
    for root_path in root_paths:
        # Create completos directory in each path
        completos_dir = os.path.join(root_path, "completos")
        os.makedirs(completos_dir, exist_ok=True)
        
        # Process all CSV files in the directory
        for filename in os.listdir(root_path):
            if filename.endswith('.csv'):
                input_file_path = os.path.join(root_path, filename)
                output_file_path = os.path.join(completos_dir, filename)
                
                try:
                    process_csv_file(input_file_path, output_file_path)
                    print(f"Processed: {input_file_path} -> {output_file_path}")
                except Exception as e:
                    print(f"Error processing {input_file_path}: {str(e)}")

if __name__ == "__main__":
    main()