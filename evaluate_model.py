import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
SAVE_DIR    = os.path.join(BASE_DIR, 'saved_model')
IMG_SIZE    = (128, 128)

model = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'banana_disease_model.keras'))

with open(os.path.join(SAVE_DIR, 'class_names.pkl'), 'rb') as f:
    class_names = pickle.load(f)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Running predictions on validation set...")
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes
num_classes = len(class_names)

per_class = {}
for i in range(num_classes):
    name = class_names[i]
    mask = y_true == i
    total = int(mask.sum())
    correct = int((y_pred[mask] == y_true[mask]).sum())
    tp = correct
    fp = int((y_pred == i).sum()) - tp
    fn = total - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    per_class[name] = {
        'accuracy':  round(correct / total * 100, 2) if total > 0 else 0.0,
        'precision': round(precision * 100, 2),
        'recall':    round(recall * 100, 2),
        'f1':        round(f1 * 100, 2),
        'samples':   total,
        'correct':   correct,
    }

overall_accuracy = round((y_pred == y_true).mean() * 100, 2)

metrics = {
    'overall_accuracy': overall_accuracy,
    'total_samples': len(y_true),
    'num_classes': num_classes,
    'model_name': 'Custom CNN (Trained from Scratch)',
    'input_size': f'{IMG_SIZE[0]} x {IMG_SIZE[1]}',
    'architecture': '4 Conv Blocks (32→64→128→256) + GAP + Dense',
    'per_class': per_class,
}

out_path = os.path.join(SAVE_DIR, 'metrics.json')
with open(out_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nOverall Accuracy: {overall_accuracy}%")
for cls, m in per_class.items():
    print(f"  {cls:20s} Acc:{m['accuracy']:6.2f}%  F1:{m['f1']:6.2f}%  ({m['samples']} samples)")
print(f"\nMetrics saved to {out_path}")