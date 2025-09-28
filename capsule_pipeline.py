
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === CONFIG ===
DATA_DIR = "/content/drive/MyDrive/capsule/classification/temp2_augmented"
IMG_SIZE = (224, 224)   # EfficientNetV2-B0 default input size
BATCH_SIZE = 16
EPOCHS = 30
MODEL_DIR = "/content/drive/MyDrive/capsule/224x224/capsule_model"  # Define model directory
MODEL_NAME = "capsnap_efficientnetv2b0_no-negative"  # Define model name


# === LOAD DATASETS ===
train_ds = image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = image_dataset_from_directory(
    f"{DATA_DIR}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# === PREFETCHING ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# === BASE MODEL ===
base_model = EfficientNetV2B0(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)

# Freeze most layers first
for layer in base_model.layers[:-30]:
    layer.trainable = False

# === MODEL WITH AUGMENTATION ===
model = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),

    # ✅ wrap preprocess_input in Lambda
    layers.Lambda(tf.keras.applications.efficientnet_v2.preprocess_input),

    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_ds.element_spec[1].shape[1], activation="softmax")
])

# === COMPILE MODEL ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# === CLASS WEIGHTS ===
labels = []
for _, y in train_ds.unbatch():
    labels.append(np.argmax(y.numpy()))
labels = np.array(labels)

class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    mode="min",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

ckpt = callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras"),
    monitor="val_accuracy",      # or "val_loss"
    save_best_only=True,
    mode="max",                  # use "min" if monitoring loss
    verbose=1
)

my_callbacks = [early_stopping, reduce_lr, ckpt]

# === TRAIN MODEL (frozen base) ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=my_callbacks,
    class_weight=class_weights
)

# === FINE-TUNE (unfreeze all) ===
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 25
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs,
    callbacks=my_callbacks,
    class_weight=class_weights
)

# === EXPORT BEST MODEL TO TFLITE ===
best_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras"))

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization (optional)
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_path}")
