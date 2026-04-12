import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

app = FastAPI(title="Classification ResNet50 - Marcel, Hugues, Jemima")

# Charger le modèle au démarrage pour gagner du temps lors des requêtes
MODEL_PATH = "app/models/mobilenetv2.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle MobileNetV2 chargé avec succès.")
except Exception as e:
    print(f"Erreur de chargement : {e}")

CLASS_NAMES = ["marcel", "hugues", "jemima"]

def preprocess_image(image_bytes):
    # Charger l'image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # ResNet50 attend généralement du 224x224
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    # Utiliser le preprocessing spécifique à ResNet50
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")
    
    contents = await file.read()
    processed_img = preprocess_image(contents)
    
    # Prédiction
    predictions = model.predict(processed_img)
    score = tf.nn.softmax(predictions[0]) # Si votre modèle ne finit pas par un softmax
    
    idx = np.argmax(predictions[0])
    label = CLASS_NAMES[idx]
    confidence = float(np.max(predictions[0])) # Ou score[idx] selon votre entraînement

    return {
        "identite": label,
        "confiance": round(confidence * 100, 2),
        "status": "success"
    }

@app.get("/")
def read_root():
    return {"message": "API de reconnaissance faciale active"}