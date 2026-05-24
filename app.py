import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

model = load_model("saved_model/best_model.keras")

class_names = [
    "Banana Xanthomonas Wilt",
    "Black Sigatoka",
    "Healthy",
    "Panama Disease",
]

def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(float(np.max(predictions)) * 100, 2)
    return {cls: float(predictions[0][i]) for i, cls in enumerate(class_names)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Banana Leaf Image"),
    outputs=gr.Label(num_top_classes=4, label="Disease Prediction"),
    title="🍌 Banana Leaf Disease Detector",
    description="Upload a banana leaf image to detect diseases using CNN model."
)

demo.launch()
