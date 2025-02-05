import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--export_directory", type=str, required=True,
                    help="Directory to export the JSON file.")
# Add other flags as needed
FLAGS, _ = parser.parse_known_args()
FLAGS = {
    "export_directory": "modeloGuardado/metadata"
}

# # Print the path to the model file
# print("Path to the model file:", quantized_model_path)

# # Load metadata from the model file
# displayer = _metadata.MetadataDisplayer.with_model_file("modeloGuardado/optimizado/quantized_model.tflite")
# export_json_file = os.path.join("modeloGuardado/metadata",
#                                 os.path.splitext("metadatosBraille")[0] + ".json")
# json_file = displayer.get_metadata_json()
# # Optional: write out the metadata as a json file
# with open(export_json_file, "w") as f:
#     f.write(json_file)