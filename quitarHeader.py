import os
import pandas as pd

def remove_header_from_csv():
    # Get current directory
    current_dir = os.getcwd()
    
    # Header to remove
    header_to_remove = "filename,label,xmin,ymin,xmax,ymax"
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(current_dir):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                
                try:
                    # Read the file content
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                    
                    # Check if the first line matches the header we want to remove
                    if lines and lines[0].strip() == header_to_remove:
                        # Write back the file without the header
                        with open(file_path, 'w') as file:
                            file.writelines(lines[1:])
                        print(f"Header removed from {file_path}")
                    else:
                        print(f"Header not found in {file_path}")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    remove_header_from_csv()