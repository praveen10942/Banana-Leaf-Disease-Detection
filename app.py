import gradio as gr
import numpy as np
import pickle
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from huggingface_hub import hf_hub_download
import tf_keras
from PIL import Image

print("Downloading model...")
model_path = hf_hub_download(
    repo_id="praveen10942/banana-disease-detector",
    filename="saved_model/banana_disease_model.h5",
    repo_type="space"
)
pkl_path = hf_hub_download(
    repo_id="praveen10942/banana-disease-detector",
    filename="saved_model/class_names.pkl",
    repo_type="space"
)

model = tf_keras.models.load_model(model_path, compile=False)

with open(pkl_path, "rb") as f:
    class_names_dict = pickle.load(f)

class_names = [class_names_dict[i] for i in range(len(class_names_dict))]

def predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

* { font-family: 'Plus Jakarta Sans', sans-serif !important; }

body, .gradio-container {
    background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%) !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Header */
#header {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    border-radius: 24px !important;
    padding: 32px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 10px 30px -5px rgba(22,163,74,0.35) !important;
    text-align: center !important;
}

/* Upload box */
.upload-box {
    border: 2px dashed #16a34a !important;
    border-radius: 20px !important;
    background: #f0fdf4 !important;
}

/* Submit button */
button.primary {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    border-radius: 14px !important;
    padding: 14px 32px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(22,163,74,0.3) !important;
    color: white !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #15803d, #166534) !important;
    transform: translateY(-2px) !important;
}

/* Output label */
.label-container {
    border-radius: 20px !important;
    border: 1px solid #d1fae5 !important;
    background: white !important;
}

.output-class {
    color: #15803d !important;
    font-weight: 800 !important;
}

.confidence-bar {
    background: linear-gradient(90deg, #10b981, #059669) !important;
    border-radius: 10px !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.emerald,
    font=gr.themes.GoogleFont("Plus Jakarta Sans")
)) as demo:

    gr.HTML("""
    <div id="header">
        <div style="font-size:2.5rem; margin-bottom:8px;">🍌</div>
        <h1 style="color:white; font-size:2rem; font-weight:800; margin:0 0 8px 0;">BananaGuard AI</h1>
        <p style="color:rgba(255,255,255,0.9); margin:0; font-size:1rem;">Smart Crop Intelligence — Powered by MobileNetV2</p>
    </div>
    """)

    gr.HTML("""
    <div style="display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap;">
        <div style="flex:1; min-width:150px; background:white; border:1px solid #d1fae5; border-radius:16px; padding:20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:1.8rem; font-weight:800; color:#16a34a;">97.2%</div>
            <div style="font-size:0.75rem; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Accuracy</div>
        </div>
        <div style="flex:1; min-width:150px; background:white; border:1px solid #d1fae5; border-radius:16px; padding:20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:1.8rem; font-weight:800; color:#16a34a;">15K+</div>
            <div style="font-size:0.75rem; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Images Trained</div>
        </div>
        <div style="flex:1; min-width:150px; background:white; border:1px solid #d1fae5; border-radius:16px; padding:20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:1.8rem; font-weight:800; color:#16a34a;">7</div>
            <div style="font-size:0.75rem; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Disease Classes</div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="📸 Upload Banana Leaf Image")
            submit_btn = gr.Button("🔍 Analyze Leaf", variant="primary")

        with gr.Column():
            output = gr.Label(num_top_classes=7, label="🧠 Disease Prediction")

    gr.HTML("""
    <div style="background:#f0fdf4; border:1px solid #d1fae5; border-radius:16px; padding:20px; margin-top:16px;">
        <h4 style="color:#15803d; margin:0 0 12px 0; font-weight:700;">✅ Detectable Conditions</h4>
        <div style="display:flex; flex-wrap:wrap; gap:10px;">
            <span style="background:white; border:1px solid #d1fae5; border-radius:20px; padding:6px 14px; font-size:0.82rem; font-weight:600; color:#374151;">✅ Healthy Leaf</span>
            <span style="background:white; border:1px solid #fef3c7; border-radius:20px; padding:6px 14px; font-size:0.82rem; font-weight:600; color:#374151;">⚠️ Black Sigatoka</span>
            <span style="background:white; border:1px solid #fef3c7; border-radius:20px; padding:6px 14px; font-size:0.82rem; font-weight:600; color:#374151;">⚠️ Yellow Sigatoka</span>
            <span style="background:white; border:1px solid #fee2e2; border-radius:20px; padding:6px 14px; font-size:0.82rem; font-weight:600; color:#374151;">🔴 Panama Disease</span>
            <span style="background:white; border:1px solid #fee2e2; border-radius:20px; padding:6px 14px; font-size:0.82rem; font-weight:600; color:#374151;">🔴 Moko Disease</span>
            <span style="background:white; border:1px solid #fef3c7; border-radius:20px; padding:6px 14px; font-size:0.82rem; font-weight:600; color:#374151;">⚠️ Bract Mosaic Virus</span>
            <span style="background:white; border:1px solid #fef3c7; border-radius:20px; padding:6px 14px; font-size:0.82rem; font-weight:600; color:#374151;">⚠️ Infectious Chlorosis</span>
        </div>
    </div>
    """)

    submit_btn.click(fn=predict, inputs=img_input, outputs=output)

demo.launch()
