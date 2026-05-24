import gradio as gr
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("saved_model/banana_disease_model.keras")

# Load class names from pkl (same ones used during training)
with open("saved_model/class_names.pkl", "rb") as f:
    class_names_dict = pickle.load(f)

# Convert {index: name} dict to ordered list
class_names = [class_names_dict[i] for i in range(len(class_names_dict))]

def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Banana Leaf Image"),
    outputs=gr.Label(num_top_classes=7, label="Disease Prediction"),
    title="🍌 Banana Leaf Disease Detector",
    description="Upload a banana leaf image to detect diseases using MobileNetV2 Transfer Learning. Detects 7 conditions."
)

demo.launch()
