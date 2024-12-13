from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import librosa
import logging
import tempfile
import os
from fastapi.middleware.gzip import GZipMiddleware
from pydub import AudioSegment

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

IMAGE_CLASSES = ['No-es-un-perro', 'Enojado', 'Feliz', 'Relajado', 'Triste']
AUDIO_CLASSES = ['Advertencia o miedo', 'Alerta o emoción', 'Relajación']

# Función para manejar archivos temporales
async def save_temp_file(file):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}")
    temp.write(await file.read())
    temp.close()
    logger.info(f'Archivo guardado en: {temp.name}')

    if not file.filename.lower().endswith("wav"):
        wav_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        convert_to_wav(temp.name, wav_file_path)  
        os.remove(temp.name)  
        logger.info(f"Archivo convertido a WAV y guardado en: {wav_file_path}")
        return wav_file_path  
    
    return temp.name  

# Función para convertir audio a formato WAV
def convert_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(22050)  
        audio.export(output_path, format="wav")
        converted_audio = AudioSegment.from_wav(output_path)
        if converted_audio.channels != 1 or converted_audio.frame_rate != 22050:
            raise ValueError(f"El archivo WAV convertido no tiene las características esperadas (canales=1, tasa de muestreo=22050).")
        logger.info(f"Archivo convertido a WAV: {output_path}")
    except Exception as e:
        logger.error(f"Error al convertir audio a WAV: {e}")
        raise HTTPException(status_code=500, detail="Error al convertir el archivo de audio")

# Modelo de imágenes
class ImageModelManager:
    def __init__(self, model_path: str = "modeloo.tflite"):
        try:
            self.model = tf.lite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()
            logger.info("Modelo de imágenes cargado correctamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo de imágenes: {e}")
            raise RuntimeError("Error al cargar el modelo de imágenes")

    def predict(self, image: Image.Image):
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        input_shape = input_details[0]['shape']

        image = image.resize((input_shape[1], input_shape[2]))
        input_array = np.array(image) / 255.0
        input_array = np.expand_dims(input_array, axis=0).astype(np.float32)

        self.model.set_tensor(input_details[0]['index'], input_array)
        self.model.invoke()
        output_data = self.model.get_tensor(output_details[0]['index'])
        logger.info("Predicción realizada con éxito")
        return output_data

# Modelo de audio
class AudioModelManager:
    def __init__(self, model_path: str = "modelo_bark.tflite"):
        try:
            self.model = tf.lite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()
            logger.info("Modelo de audio cargado correctamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo de audio: {e}")
            raise RuntimeError("Error al cargar el modelo de audio")

    def preprocess_audio(self, file_path, sr=22050, duration=5):
        audio, _ = librosa.load(file_path, sr=sr)
        target_length = sr * duration
        return librosa.util.fix_length(audio, size=target_length)

    def extract_features(self, audio, sr=22050, n_mfcc=13):
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)

    def predict(self, file_path):
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        audio = self.preprocess_audio(file_path)
        features = self.extract_features(audio)
        features = np.expand_dims(features, axis=0)

        logger.info(f"Forma de características ajustadas: {features.shape}")
        logger.info(f"Forma esperada por el modelo: {input_details[0]['shape']}")

        self.model.set_tensor(input_details[0]['index'], features)
        self.model.invoke()
        output_data = self.model.get_tensor(output_details[0]['index'])
        logger.info("Predicción realizada con éxito")
        return output_data

# Inicializar modelos
image_model_manager = ImageModelManager()
audio_model_manager = AudioModelManager()

def interpret_prediction(prediction):
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    return {"class": IMAGE_CLASSES[class_idx], "confidence": float(confidence)}

def interpretar_audio(prediction):
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    return {"class": AUDIO_CLASSES[class_idx], "confidence": float(confidence)}

@app.get("/")
def inicio():
    return {"message": "Hola mundo, conexión correctamente realizada"}

@app.post("/predict/image/", summary="Predice clases de imágenes", description="Sube una imagen para predecir.")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(("jpg", "jpeg", "png")):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen en formato JPG o PNG")
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        result = image_model_manager.predict(image)
        logger.info(f"Resultado del modelo de imagen: {result}")
        interpreted = interpret_prediction(result[0]) 
        logger.info(f"resultado Interpretado: {interpreted}")
        return {"prediction": interpreted}    
    except Exception as e:
        logger.error(f"Error al procesar imagen: {e}")
        raise HTTPException(status_code=500, detail="Error interno en predicción de imagen")
    finally:
        image.close()

@app.post("/predict/audio/", summary="Predice clases de audio", description="Sube un archivo de audio para predecir.")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(("wav", "mp3", "m4a")):
        raise HTTPException(status_code=400, detail="El archivo debe ser un audio en formato WAV o MP3")
    try:
        file_path = await save_temp_file(file)
        result = audio_model_manager.predict(file_path)
        interpretado = interpretar_audio(result[0])
        logger.info(f"resultado Interpretado: {interpretado}")
        return {"prediction": interpretado}
    except Exception as e:
        logger.error(f"Error al procesar audio: {e}")
        raise HTTPException(status_code=500, detail="Error interno en predicción de audio")
    finally:
        os.remove(file_path)
