import gradio as gr
import numpy as np
import pickle
import os
import subprocess
from tensorflow.keras.models import load_model
from PIL import Image

# Pull actual LFS files (replaces pointer files with real content)
print("Running git lfs pull...")
result = subprocess.run(
    ["git", "lfs", "pull"],
    cwd="/app",
    capture_output=True,
    text=True
)
print("LFS stdout:", result.stdout)
print("LFS stderr:", result.stderr)

# Load model
model_path = "/app/saved_model/best_model.keras"
h5_path = "/app/saved_model/banana_disease_model.h5"

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = load_model(h5_path)

# Load class names
with open("/app/saved_model/class_names.pkl", "rb") as f:
    class_names_dict = pickle.load(f)

class_names = [class_names_dict[i] for i in range(len(class_names_dict))]

def predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Banana Leaf Image"),
    outputs=gr.Label(num_top_classes=7, label="Disease Prediction"),
    title="🍌 Banana Leaf Disease Detector",
    description="Upload a banana leaf image to detect diseases using MobileNetV2 Transfer Learning."
)

demo.launch()
