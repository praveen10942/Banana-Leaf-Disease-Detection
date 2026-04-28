import os
import json
import pickle
import numpy as np
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.db.models import Count
from .models import Prediction

MODEL_DIR  = os.path.join(settings.BASE_DIR, 'saved_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'banana_disease_model.keras')
NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.pkl')

model       = None
class_names = None

def load_model():
    global model, class_names
    if model is None:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(NAMES_PATH, 'rb') as f:
                class_names = pickle.load(f)
        except Exception as e:
            print(f"Model load failed: {e}")

load_model()

DISEASE_INFO = {
    'healthy': {
        'label': 'Healthy',
        'description': 'The leaf appears healthy with no visible signs of infection or disease.',
        'action': 'Continue regular care, watering and monitoring.',
        'severity': 'None',
        'color': 'success',
    },
    'sigatoka': {
        'label': 'Sigatoka',
        'description': 'Fungal leaf spot disease caused by Mycosphaerella musicola. Shows yellow streaks and brown spots on leaf surface.',
        'action': 'Apply copper-based fungicide. Remove heavily infected leaves. Improve field drainage.',
        'severity': 'Moderate',
        'color': 'warning',
    },
    'cordana': {
        'label': 'Cordana Leaf Spot',
        'description': 'Fungal disease caused by Cordana musae. Appears as oval brown spots with yellow halos on leaves.',
        'action': 'Apply fungicide spray. Remove and destroy infected leaves. Avoid overhead irrigation.',
        'severity': 'Moderate',
        'color': 'warning',
    },
    'pestalotiopsis': {
        'label': 'Pestalotiopsis',
        'description': 'Fungal disease caused by Pestalotiopsis species. Causes dark necrotic spots along leaf edges.',
        'action': 'Remove infected leaves immediately. Apply systemic fungicide. Improve air circulation around plants.',
        'severity': 'High',
        'color': 'danger',
    },
}


def preprocess_image(image_path):
    import tensorflow as tf
    img = tf.keras.utils.load_img(image_path, target_size=(128, 128))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def landing(request):
    if request.user.is_authenticated:
        return render(request, 'predictor/landing.html')
    return render(request, 'predictor/landing.html')


@login_required
def upload(request):
    if request.method == 'POST':
        if 'image' not in request.FILES:
            messages.error(request, 'Please select an image.')
            return redirect('upload')

        img_file = request.FILES['image']

        if img_file.content_type not in ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']:
            messages.error(request, 'Only JPG, PNG or WEBP images are supported.')
            return redirect('upload')

        if model is None:
            messages.error(request, 'Model is not loaded. Please run train_cnn.py first.')
            return redirect('upload')

        pred_obj = Prediction(user=request.user, image=img_file, disease='Pending', confidence=0.0)
        pred_obj.save()

        img_path  = os.path.join(settings.MEDIA_ROOT, pred_obj.image.name)
        input_arr = preprocess_image(img_path)
        preds     = model.predict(input_arr, verbose=0)[0]

        top_idx   = int(np.argmax(preds))
        top_conf  = float(preds[top_idx]) * 100
        raw_class = class_names[top_idx]

        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [
            {
                'label': DISEASE_INFO.get(class_names[i], {}).get('label', class_names[i].title()),
                'confidence': round(float(preds[i]) * 100, 1),
                'color': DISEASE_INFO.get(class_names[i], {}).get('color', 'secondary'),
            }
            for i in top3_idx
        ]

        info = DISEASE_INFO.get(raw_class, {
            'label': raw_class.title(),
            'description': 'Unknown condition detected.',
            'action': 'Please consult an agronomist.',
            'severity': 'Unknown',
            'color': 'secondary',
        })

        pred_obj.disease    = info['label']
        pred_obj.confidence = round(top_conf, 1)
        pred_obj.is_healthy = (raw_class == 'healthy')
        pred_obj.save()

        return render(request, 'predictor/result.html', {
            'pred': pred_obj,
            'info': info,
            'top3': top3,
            'raw_conf': round(top_conf, 1),
        })

    return render(request, 'predictor/upload.html')


@login_required
def history(request):
    filter_type = request.GET.get('filter', 'all')
    qs = Prediction.objects.filter(user=request.user)
    if filter_type == 'healthy':
        qs = qs.filter(is_healthy=True)
    elif filter_type == 'diseased':
        qs = qs.filter(is_healthy=False)
    preds = qs[:30]
    return render(request, 'predictor/history.html', {'preds': preds, 'filter_type': filter_type})


@login_required
def dashboard(request):
    total    = Prediction.objects.filter(user=request.user).count()
    healthy  = Prediction.objects.filter(user=request.user, is_healthy=True).count()
    diseased = total - healthy
    recent   = Prediction.objects.filter(user=request.user)[:5]
    disease_counts = (
        Prediction.objects
        .filter(user=request.user, is_healthy=False)
        .values('disease')
        .annotate(count=Count('disease'))
        .order_by('-count')
    )
    return render(request, 'predictor/dashboard.html', {
        'total': total,
        'healthy': healthy,
        'diseased': diseased,
        'recent': recent,
        'disease_counts': disease_counts,
    })


def about(request):
    return render(request, 'predictor/about.html')


def model_stats(request):
    metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    return render(request, 'predictor/model_stats.html', {'metrics': metrics})