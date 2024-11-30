from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Clases del modelo (ajusta según tu caso)
CLASSES = ['angry', 'happy', 'relaxed', 'sad']

# Clase para gestionar el modelo
class ModelManager:
    def __init__(self):
        try:
            # Cargar el modelo al iniciar el servidor
            self.model = tf.lite.Interpreter(model_path="modelo.tflite")
            self.model.allocate_tensors()
            logger.info("Modelo cargado correctamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise RuntimeError("Error al cargar el modelo")

    def predict(self, image: Image.Image):
        if self.model is None:
            raise RuntimeError("El modelo no está cargado.")

        try:
            # Obtener detalles del input del modelo
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()

            # Validar dimensiones del input
            input_shape = input_details[0]['shape']
            image = image.resize((input_shape[1], input_shape[2]))

            # Normalizar la imagen y convertirla en un array
            input_array = np.array(image) / 255.0
            input_array = np.expand_dims(input_array, axis=0).astype(np.float32)

            # Realizar la predicción
            self.model.set_tensor(input_details[0]['index'], input_array)
            self.model.invoke()
            output_data = self.model.get_tensor(output_details[0]['index'])

            logger.info("Predicción realizada con éxito")
            return output_data
        except Exception as e:
            logger.error(f"Error al realizar la predicción: {e}")
            raise RuntimeError("Error al realizar la predicción")

# Instancia del gestor del modelo
model_manager = ModelManager()

# Función para interpretar la predicción
def interpret_prediction(prediction):
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    return {"class": CLASSES[class_idx], "confidence": float(confidence)}

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/predict/", summary="Realiza una predicción", description="Sube una imagen y usa el modelo cargado para predecir.")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(("jpg", "jpeg", "png")):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen en formato JPG o PNG")

    try:
        # Leer y procesar la imagen
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        prediction = model_manager.predict(image)
        interpreted = interpret_prediction(prediction[0])  # Asegurarse de pasar solo la primera dimensión
        return {"prediction": interpreted}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {str(e)}")

