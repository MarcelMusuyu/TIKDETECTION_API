from pathlib import Path
import cv2
import gradio as gr
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "app/models/best.pt"
CONF_THRESHOLD = 0.25
IMG_SIZE = 640

# =========================================================
# CHARGEMENT MODELE
# =========================================================
model_path = Path(MODEL_PATH)
if not model_path.exists():
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

model = YOLO(str(model_path))
print("Classes du modèle :", model.names)

# =========================================================
# OUTILS
# =========================================================
def predict_image(image, conf_threshold):
    if image is None:
        return None, "Aucune image reçue."

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=image_bgr,
        conf=float(conf_threshold),
        imgsz=IMG_SIZE,
        verbose=False
    )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if result.boxes is None or len(result.boxes) == 0:
        return annotated_rgb, "Aucune détection."

    lines = [f"Nombre de détections : {len(result.boxes)}", ""]

    for i, box in enumerate(result.boxes, start=1):
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

        class_name = model.names[cls_id]
        lines.append(
            f"Détection {i} : {class_name} | confiance={conf:.3f} | bbox={xyxy}"
        )

    return annotated_rgb, "\n".join(lines)

# =========================================================
# INTERFACE
# =========================================================
with gr.Blocks(title="Test YOLO - 3 personnes") as demo:
    gr.Markdown("# Test du modèle YOLO")
    gr.Markdown(
        "Tu peux uploader une image ou prendre une photo avec la webcam "
        "pour détecter les 3 personnes du modèle."
    )

    with gr.Row():
        image_input = gr.Image(
            type="numpy",
            sources=["upload", "webcam"],
            label="Image d'entrée"
        )

        image_output = gr.Image(
            type="numpy",
            label="Résultat annoté"
        )

    conf_slider = gr.Slider(
        minimum=0.05,
        maximum=0.95,
        value=CONF_THRESHOLD,
        step=0.05,
        label="Seuil de confiance"
    )

    predict_btn = gr.Button("Lancer la prédiction")

    result_text = gr.Textbox(
        label="Détails des détections",
        lines=10
    )

    predict_btn.click(
        fn=predict_image,
        inputs=[image_input, conf_slider],
        outputs=[image_output, result_text]
    )

# =========================================================
# LANCEMENT
# =========================================================
if __name__ == "__main__":
    demo.launch()