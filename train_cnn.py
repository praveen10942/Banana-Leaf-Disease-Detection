import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
SAVE_DIR    = os.path.join(BASE_DIR, 'saved_model')
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 40
os.makedirs(SAVE_DIR, exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
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

print(f"Classes: {list(train_gen.class_indices.keys())}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

total_samples = train_gen.samples
class_counts = {}
for cls, idx in train_gen.class_indices.items():
    cls_path = os.path.join(DATASET_DIR, cls)
    count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
    class_counts[idx] = count

print("Samples per class:")
for cls, idx in train_gen.class_indices.items():
    print(f"  {cls}: {class_counts[idx]}")

class_weights = {
    idx: total_samples / (len(class_counts) * count)
    for idx, count in class_counts.items()
}
print("Class weights applied:")
for cls, idx in train_gen.class_indices.items():
    print(f"  {cls}: {class_weights[idx]:.3f}")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint(
        os.path.join(SAVE_DIR, 'banana_disease_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\nTraining custom CNN from scratch...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

val_gen.reset()
y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
y_true = val_gen.classes

overall_acc = round((y_pred == y_true).mean() * 100, 2)
print(f"\nFinal Validation Accuracy: {overall_acc}%")

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
    print(f"  {name:20s} Acc:{per_class[name]['accuracy']:6.2f}%  F1:{per_class[name]['f1']:6.2f}%  ({total} samples)")

metrics = {
    'overall_accuracy': overall_acc,
    'total_samples': len(y_true),
    'num_classes': num_classes,
    'model_name': 'Custom CNN (Trained from Scratch)',
    'input_size': f'{IMG_SIZE[0]} x {IMG_SIZE[1]}',
    'architecture': '4 Conv Blocks (32→64→128→256) + GAP + Dense',
    'per_class': per_class,
}

with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nModel saved to saved_model/banana_disease_model.keras")
print(f"Metrics saved to saved_model/metrics.json")
print("Training complete.")