import csv
import os
import re

def convert_number_to_letter(number):
    """
    Converts a number to its corresponding letter based on the given encode table.

    Args:
        number: The number to be converted.

    Returns:
        The corresponding letter based on the encode table.
    """
    encode_table = {
        0: ' ',
        1: 'A',
        2: '1',
        3: 'B',
        4: "'",
        5: 'K',
        6: '2',
        7: 'L',
        8: '3',
        9: 'C',
        10: 'I',
        11: 'F',
        12: '/',
        13: 'M',
        14: 'S',
        15: 'P',
        16: '"',
        17: 'E',
        18: '3',
        19: 'H',
        20: '9',
        21: 'O',
        22: '6',
        23: 'R',
        24: '^',
        25: 'D',
        26: 'J',
        27: 'G',
        28: '>',
        29: 'N',
        30: 'T',
        31: 'Q',
        32: ',',
        33: '*',
        34: '5',
        35: '-',
        36: '<',
        37: 'U',
        38: '8',
        39: 'V',
        40: '.',
        41: '%',
        42: '$',
        43: '+',
        44: 'X',
        45: '!',
        46: '&',
        47: ';',
        48: ':',
        49: '4',
        50: '0',
        51: '[',
        52: ']',
        53: '8',
        54: '(',
        55: ')',
        56: 'W',
        57: '7',
        58: '2',
        59: '1',
        60: 'Z',
        61: 'Y',
        62: '=',
        63: '6'
    }

    return encode_table.get(number, '')

def contains_only_letters(s):
    """
    Checks if a string contains only letters.

    Args:
        s: The string to be checked.

    Returns:
        True if the string contains only letters, False otherwise.
    """
    return bool(re.match(r'^[A-Za-z]+$', s))

def generate_csv(folder_path, output_folder):
    """
    Generates CSV files from a folder containing JPEGs and corresponding CSVs, assigning labels based on another CSV file.

    Args:
        folder_path: Path to the folder containing JPEGs and CSVs.
        output_folder: Path to the folder where output CSV files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(folder_path, filename)
            image_path = filepath.replace("\\", "/")  # Replace backslashes with forward slashes

            # Extract label and other values from corresponding CSV file
            csv_filename = os.path.splitext(filename)[0] + ".csv"
            csv_filepath = os.path.join(folder_path, csv_filename)
            if not os.path.exists(csv_filepath):
                print(f"CSV file {csv_filepath} does not exist. Skipping.")
                continue

            data = []
            with open(csv_filepath, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=';')  # Specify delimiter
                next(csvreader)  # Skip header row
                for row in csvreader:
                    if len(row) < 5:  # Ensure that there are enough columns in the row
                        print(f"Skipping {csv_filename} row due to insufficient columns.")
                        continue
                    label = convert_number_to_letter(int(row[4]))  # Convert number to letter based on encode table
                    other_values = [row[i] for i in range(4)]  # Extract first four columns
                    data.append([image_path, label] + other_values)  # Append data to list

            # Remove rows where the label contains numbers or special characters
            data = [row for row in data if contains_only_letters(row[1])]

            # Write data to CSV file
            output_filename = os.path.join(output_folder, os.path.splitext(filename)[0] + "_traducido.csv")
            with open(output_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # csvwriter.writerow(['Label', 'Image Path', '1st Column', '2nd Column', '3rd Column', '4th Column'])
                csvwriter.writerows(data)

            print(f"CSV file generated successfully: {output_filename}")

# List of folder paths
folder_paths = [
    "datasets/AngelinaDataset/books/chudo_derevo_redmi",
    "datasets/AngelinaDataset/books/mdd_cannon1",
    "datasets/AngelinaDataset/books/mdd-redmi1",
    "datasets/AngelinaDataset/books/ola",
    "datasets/AngelinaDataset/books/skazki",
    "datasets/AngelinaDataset/books/telefon",
    "datasets/AngelinaDataset/books/uploaded",
    "datasets/AngelinaDataset/handwritten/ang_redmi",
    "datasets/AngelinaDataset/handwritten/kov",
    "datasets/AngelinaDataset/handwritten/uploaded"
]

# Execute generate_csv for each folder path
for folder_path in folder_paths:
    output_folder = folder_path + "/traducido"
    generate_csv(folder_path, output_folder)
