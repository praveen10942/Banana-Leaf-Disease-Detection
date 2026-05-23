import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
SAVE_DIR    = os.path.join(BASE_DIR, 'saved_model')
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
EPOCHS      = 60
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Data Augmentation ──────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=30,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_gen.class_indices)
class_names = {v: k for k, v in train_gen.class_indices.items()}

with open(os.path.join(SAVE_DIR, 'class_names.pkl'), 'wb') as f:
    pickle.dump(class_names, f)

print(f"\n{'='*60}")
print(f"  Banana Leaf Disease Detection — Training")
print(f"  Architecture: MobileNetV2 (Transfer Learning)")
print(f"  Image Size:   {IMG_SIZE[0]}x{IMG_SIZE[1]}")
print(f"  Classes:      {num_classes}")
print(f"{'='*60}")
print(f"\nClasses: {list(train_gen.class_indices.keys())}")
print(f"Training samples:   {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# ── Class weights (handle imbalanced data) ────────────────────────────
total_samples = train_gen.samples
class_counts = {}
for cls, idx in train_gen.class_indices.items():
    cls_path = os.path.join(DATASET_DIR, cls)
    count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
    class_counts[idx] = count

print("\nSamples per class:")
for cls, idx in train_gen.class_indices.items():
    print(f"  {cls:30s}: {class_counts[idx]} images")

class_weights = {
    idx: total_samples / (len(class_counts) * count)
    for idx, count in class_counts.items()
}
print("\nClass weights applied:")
for cls, idx in train_gen.class_indices.items():
    print(f"  {cls:30s}: {class_weights[idx]:.3f}")

# ── Model: MobileNetV2 Transfer Learning ──────────────────────────────
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

# Freeze base model initially
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nTotal params:     {model.count_params():,}")
print(f"Trainable params: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint(
        os.path.join(SAVE_DIR, 'banana_disease_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ── Phase 1: Train top layers only ────────────────────────────────────
print("\n" + "="*60)
print("  Phase 1: Training top layers (base frozen)")
print("="*60)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ── Phase 2: Fine-tune last 30 layers of MobileNetV2 ─────────────────
print("\n" + "="*60)
print("  Phase 2: Fine-tuning last 30 layers")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Trainable params after unfreeze: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    initial_epoch=len(history1.history['loss']),
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ── Evaluation ────────────────────────────────────────────────────────
val_gen.reset()
y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
y_true = val_gen.classes

overall_acc = round((y_pred == y_true).mean() * 100, 2)
print(f"\n{'='*60}")
print(f"  Final Validation Accuracy: {overall_acc}%")
print(f"{'='*60}")

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
    print(f"  {name:30s} Acc:{per_class[name]['accuracy']:6.2f}%  F1:{per_class[name]['f1']:6.2f}%  ({total} samples)")

metrics = {
    'overall_accuracy': overall_acc,
    'total_samples': len(y_true),
    'num_classes': num_classes,
    'model_name': 'MobileNetV2 (Transfer Learning)',
    'input_size': f'{IMG_SIZE[0]} x {IMG_SIZE[1]}',
    'architecture': 'MobileNetV2 + GAP + Dense(512) + Dense(256) + Softmax',
    'training_phases': 'Phase 1: Frozen base → Phase 2: Fine-tune last 30 layers',
    'per_class': per_class,
}

with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nModel saved to saved_model/banana_disease_model.keras")
print(f"Metrics saved to saved_model/metrics.json")
print("Training complete.")