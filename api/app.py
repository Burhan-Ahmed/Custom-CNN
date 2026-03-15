from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Alzheimer MRI Classification API")

# Load trained model
model = tf.keras.models.load_model("model/alzheimer_cnn_model.h5")

# Class labels
classes = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

# Image preprocessing
def preprocess_image(image):

    image = image.resize((224,224))
    image = np.array(image) / 255.0

    if image.shape[-1] == 4:
        image = image[:,:,:3]

    image = np.expand_dims(image, axis=0)

    return image


@app.get("/")
def home():
    return {"message": "Alzheimer MRI Prediction API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {
        "model": "Alzheimer CNN",
        "classes": classes,
        "input_size": "224x224"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed = preprocess_image(image)

    prediction = model.predict(processed)

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "prediction": predicted_class,
        "confidence": confidence
    }
