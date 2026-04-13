import io
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

# =========================================================
# CONFIGURATION & CHARGEMENT DES MODÈLES
# =========================================================

app = FastAPI(
    title="Face Recognition & Detection API",
    description="API combinée : Classification (MobileNetV2) et Détection (YOLO)",
    version="1.0.0"
)

# Chemins des modèles
YOLO_MODEL_PATH = "app/models/best.pt"
K_MODEL_PATH = "app/models/mobilenetv2.keras"

# Chargement YOLO
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ Modèle YOLO chargé.")
except Exception as e:
    print(f"❌ Erreur YOLO : {e}")
    yolo_model = None

# Chargement MobileNetV2
try:
    classifier_model = tf.keras.models.load_model(K_MODEL_PATH)
    print("✅ Modèle MobileNetV2 chargé.")
except Exception as e:
    print(f"❌ Erreur MobileNetV2 : {e}")
    classifier_model = None

CLASS_NAMES = ["marcel", "hugues", "jemima"]

# =========================================================
# UTILITAIRES
# =========================================================

def preprocess_for_classifier(image_bytes):
    """Prépare l'image pour le modèle de classification."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    # Note : MobileNetV2 a son propre preprocess_input, 
    # mais si vous avez entraîné avec 1/255.0 :
    img_array = img_array / 255.0 
    return np.expand_dims(img_array, axis=0)

# =========================================================
# ROUTES API
# =========================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "yolo_ready": yolo_model is not None,
        "classifier_ready": classifier_model is not None
    }

@app.post("/classify")
async def classify_face(file: UploadFile = File(...)):
    """Classification simple (Qui est-ce sur l'image ?)"""
    if classifier_model is None:
        raise HTTPException(status_code=500, detail="Modèle de classification non disponible.")
    
    contents = await file.read()
    processed_img = preprocess_for_classifier(contents)
    
    predictions = classifier_model.predict(processed_img)
    idx = np.argmax(predictions[0])
    
    return {
        "type": "classification",
        "identite": CLASS_NAMES[idx],
        "confiance": float(np.max(predictions[0])),
        "scores": {name: float(prob) for name, prob in zip(CLASS_NAMES, predictions[0])}
    }

@app.post("/detect")
async def detect_people(file: UploadFile = File(...), conf: float = 0.25):
    """Détection d'objets (Où sont-ils sur l'image ?)"""
    if yolo_model is None:
        raise HTTPException(status_code=500, detail="Modèle YOLO non disponible.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = yolo_model.predict(source=image_bgr, conf=conf, imgsz=640, verbose=False)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            detections.append({
                "label": yolo_model.names[cls_id],
                "confidence": round(float(box.conf[0].cpu().numpy()), 4),
                "bbox": box.xyxy[0].cpu().numpy().tolist()
            })

        return {
            "type": "detection",
            "count": len(detections),
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))