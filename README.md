---
title: Banana Disease Detector
emoji: 🍌
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
---







# 🍌 Banana Leaf Disease Detector (CNN)

An end-to-end deep learning web app built with **Django** and **MobileNetV2** (Transfer Learning) that detects banana leaf diseases from uploaded photos.

## Detects 7 Conditions
- ✅ Healthy
- ⚠️ Black Sigatoka
- ⚠️ Yellow Sigatoka
- ⚠️ Panama Disease
- ⚠️ Moko Disease
- ⚠️ Bract Mosaic Virus
- ⚠️ Infectious Chlorosis

## Tech Stack
- **Backend:** Django 4.2
- **Deep Learning:** TensorFlow / Keras — MobileNetV2 (Transfer Learning)
- **Frontend:** Bootstrap 5 (minimal UI)
- **Database:** SQLite (stores prediction history per user)

---

## Setup Instructions

### 1. Create and activate virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset from Kaggle
👉 https://www.kaggle.com/datasets/rishabhrpandey/banana-leaf-disease-dataset

Extract it so your `dataset/` folder looks like:
```
dataset/
  Banana_Healthy_Leaf/
  Banana_Black_Sigatoka_Disease/
  Banana_Yellow_Sigatoka_Disease/
  Banana_Panama_Disease/
  Banana_Moko_Disease/
  Banana_Bract_Mosaic_Virus_Disease/
  Banana_Infectious_Chlorosis_Disease/
```

### 4. Train the CNN model
```bash
python train_cnn.py
```
This saves `saved_model/banana_disease_model.keras` and `saved_model/class_names.pkl`.
Training takes ~10–30 minutes depending on your hardware. GPU recommended.

### 5. Apply migrations
```bash
python manage.py migrate
```

### 6. Run the server
```bash
python manage.py runserver
```

Open http://127.0.0.1:8000 — register an account and start uploading leaf images.

---

## Project Structure
```
banana_cnn_project/
├── train_cnn.py              ← Train MobileNetV2 on your dataset
├── manage.py
├── requirements.txt
├── dataset/                  ← Put Kaggle dataset here
├── saved_model/              ← Auto-generated after training
│   ├── banana_disease_model.keras
│   └── class_names.pkl
├── media/uploads/            ← Uploaded images (auto-created)
├── leaf_project/             ← Django project config
├── predictor/                ← Main ML app (upload, result, history)
├── users/                    ← Auth app (register, login, logout)
└── templates/                ← Shared base template
```

## Model Architecture
- **Base:** MobileNetV2 pretrained on ImageNet (frozen)
- **Head:** GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.4) → Dense(7, Softmax)
- **Phase 1:** Train head only (10 epochs)
- **Phase 2:** Fine-tune top 30 layers of MobileNetV2 (20 epochs)
- **Expected accuracy:** 95%+







