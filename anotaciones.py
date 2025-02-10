import pandas as pd
import os
import numpy as np

# List of directories containing CSV files
csv_directories = [
    'datasets/AngelinaDataset/books/chudo_derevo_redmi/traducido/filtros/anotaciones'
    # Add more directories as needed
]

# List of individual CSV file paths
csv_files = [
    'datasets/libroINCI/datasetprueba1FILTROS/anotaciones/datasetprueba1FILTROS.csv'
    # Add more individual files as needed
]

# Add all CSV files from the directories to the csv_files list
for directory in csv_directories:
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(directory, file))

# Create an empty list to hold the dataframes
dataframes = []

# Read each CSV file and append the dataframe to the list
for file in csv_files:
    df = pd.read_csv(file, header=None)
    dataframes.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Create a new column with random assignment of 'training', 'test', 'validation'
np.random.seed(42)  # For reproducibility
labels = np.random.choice(['TRAIN', 'VAL', 'TEST'], size=len(combined_df), p=[0.8, 0.1, 0.1])
combined_df.insert(0, 'set', labels)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output file path
output_file = os.path.join(script_dir, 'anotaciones.csv')

# Save the combined dataframe to a CSV file
combined_df.to_csv(output_file, index=False, header=False)

print(f'Combined CSV saved to {output_file}')