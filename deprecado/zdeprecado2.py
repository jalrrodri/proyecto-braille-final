# Importa Path para trabajar con rutas de archivos
from pathlib import Path
import os

# Especifica la ruta al conjunto de datos de imágenes
data_dir = Path("images")

# Importar metadatos
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

quantized_model_path = "modeloGuardado/optimizado/quantized_model.tflite"

# Cargar el modelo cuantizado
try:
    with open(quantized_model_path, "rb") as f:
        quantized_model_content = f.read()
except Exception as e:
    print("Error cargando el modelo cuantizado:", e)

# Definir información del modelo para metadatos
try:
    # Crea información del modelo.
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "Braille"
    model_meta.description = ("Identifica el objeto más prominente en la "
                                "imagen de un conjunto de 1,001 categorías como "
                                "árboles, animales, comida, vehículos, persona, etc.")
    model_meta.version = "v1"
    model_meta.author = "JARR JSHS"
    model_meta.license = ("Licencia Apache. Versión 2.0 "
                        "http://www.apache.org/licenses/LICENSE-2.0.")
except Exception as e:
    print("Error al definir los metadatos del modelo:", e)

# Definir información de entrada para metadatos
input_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "image"
input_meta.description = (
    "Imagen de entrada a clasificar. La imagen esperada es de {0} x {1}, con "
    "tres canales (rojo, azul y verde) por píxel. Cada valor en el "
    "tensor es un solo byte entre 0 y 255.".format(160, 160))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (_metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (_metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [200.80178817, 200.80178817, 200.80178817]  # Usar los valores medios calculados
input_normalization.options.std = [72.94811093, 72.94811093, 72.94811093]  # Usar los valores de desviación estándar calculados
input_meta.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [0]
input_meta.stats = input_stats

# Definir información de salida para metadatos
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "resultado_deteccion"
output_meta.description = "Caracteres Braille detectados."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.FeatureProperties)
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats

try:
    # Crear información de subgráfico
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]
except Exception as e:
    print("Error creando información de subgráfico:", e)

try:
    b = flatbuffers.Builder()
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()
except Exception as e:
    print("Error creando los Flatbuffers de los metadatos:", e)

try:
    # Empaquetar metadatos en el modelo
    populator = _metadata.MetadataPopulator.with_model_file(quantized_model_path)
    populator.load_metadata_buffer(metadata_buf)
    populator.populate()
    # Imprimir el tamaño del búfer del modelo después de poblar metadatos
    print("Tamaño del búfer del modelo después de poblar metadatos:", len(populator.get_model_buffer()))
except Exception as e:
    print("Error empaquetando el modelo con metadatos:", e)

try:
    # ...
    model_buffer = populator.get_model_buffer()
    if not model_buffer:
        raise Exception("Error: Búfer del modelo vacío después de empaquetar metadatos")
    with open(quantized_model_path, "wb") as f:
        f.write(model_buffer)
        if os.path.getsize(quantized_model_path) > 0:
            print("El modelo con metadatos se ha guardado en:", str(quantized_model_path))
        else:
            raise Exception("Error: El modelo con metadatos no se guardó correctamente.")
except Exception as e:
    print("Error guardando el modelo con metadatos:", e)
    # Manejar error (por ejemplo, lanzar excepción, reintentar)
