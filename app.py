import gradio as gr
import numpy as np
import pickle
import os
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image

print("Downloading model from HuggingFace Hub...")

model_path = hf_hub_download(
    repo_id="praveen10942/banana-disease-detector",
    filename="saved_model/best_model.keras",
    repo_type="space"
)
print(f"Model downloaded to: {model_path}")

pkl_path = hf_hub_download(
    repo_id="praveen10942/banana-disease-detector",
    filename="saved_model/class_names.pkl",
    repo_type="space"
)

model = load_model(model_path)
print("Model loaded successfully!")

with open(pkl_path, "rb") as f:
    class_names_dict = pickle.load(f)

class_names = [class_names_dict[i] for i in range(len(class_names_dict))]
print(f"Classes: {class_names}")

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
