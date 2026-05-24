import gradio as gr
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from PIL import Image

# Debug: print all files
print("=== Files in /app/ ===")
for root, dirs, files in os.walk("/app"):
    for f in files:
        print(os.path.join(root, f))

# Load model using absolute path
model_path = "/app/saved_model/best_model.keras"
h5_path = "/app/saved_model/banana_disease_model.h5"

if os.path.exists(model_path):
    print(f"Loading: {model_path}")
    model = load_model(model_path)
elif os.path.exists(h5_path):
    print(f"Loading: {h5_path}")
    model = load_model(h5_path)
else:
    raise FileNotFoundError(f"No model found! Searched: {model_path}, {h5_path}")

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
