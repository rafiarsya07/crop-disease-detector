import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pathlib

MODEL_PATH  = "crop_disease_model.h5"
LABELS_PATH = "class_labels.json"
IMG_SIZE    = (224, 224)

def load_model():
    if not pathlib.Path(MODEL_PATH).exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return model, labels

model, labels = load_model()

def parse_label(label):
    """Parse label format apapun jadi plant + condition"""
    # Format 1: Tomato___Late_blight
    if "___" in label:
        parts = label.split("___")
        plant = parts[0].replace("_", " ").strip()
        condition = parts[1].replace("_", " ").strip()
    # Format 2: Tomato_Late_blight atau Tomato healthy
    elif "_" in label:
        parts = label.split("_")
        plant = parts[0].strip()
        condition = " ".join(parts[1:]).strip()
    else:
        plant = label
        condition = "Unknown"
    
    # Kalau condition kosong
    if not condition:
        condition = "Unknown"
    
    is_healthy = "healthy" in condition.lower()
    return plant, condition, is_healthy

def predict(image):
    if model is None:
        return "❌ Model not found."
    if image is None:
        return "Please upload an image."

    img = Image.fromarray(image).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]
    top_indices = predictions.argsort()[-3:][::-1]

    results = ""
    for i, idx in enumerate(top_indices):
        label = labels[str(idx)]
        confidence = float(predictions[idx] * 100)
        plant, condition, is_healthy = parse_label(label)

        if i == 0:
            status = "✅ Healthy" if is_healthy else "⚠️ Diseased"
            results += f"## {status}\n\n"
            results += f"**Plant:** {plant}\n\n"
            results += f"**Condition:** {condition}\n\n"
            results += f"**Confidence:** {confidence:.1f}%\n\n"
            results += "---\n\n**Other possibilities:**\n\n"
        else:
            status_icon = "✅" if is_healthy else "⚠️"
            results += f"- {status_icon} {plant} — {condition} ({confidence:.1f}%)\n"

    results += "\n\n---\n*This is an AI prediction. Consult an agricultural expert for certainty.*"
    return results

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload leaf photo"),
    outputs=gr.Markdown(label="Result"),
    title="🌿 Crop Disease Detector",
    description="Upload a leaf photo to detect plant diseases using AI.\nBuilt with Python + TensorFlow | Dataset: PlantVillage | Model: MobileNetV2",
    theme=gr.themes.Soft()
)

demo.launch()
