from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === CONFIG ===
DATA_DIR = "/content/drive/MyDrive/Classification of Crab/temp2/crab_6_dataset"
IMG_SIZE = (480, 480)   # EfficientNetV2-M default input size
BATCH_SIZE = 16
EPOCHS = 50
MODEL_DIR = "/content/drive/MyDrive/Classification of Crab/models"
MODEL_NAME = "crab_efficientnetv2m_class6_temp2"

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
base_model = EfficientNetV2M(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)

# Freeze most layers initially
for layer in base_model.layers[:-30]:
    layer.trainable = False

# === MODEL WITH AUGMENTATION ===
model = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),

    layers.Lambda(tf.keras.applications.efficientnet_v2.preprocess_input),

    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.4),
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


### === CALLBACKS ===
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
    mode="max",                  # use "min" if monitoring val_loss
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


# === PLOTTING ===
def plot_history(history, fine_tune_history=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if fine_tune_history:
        acc += fine_tune_history.history['accuracy']
        val_acc += fine_tune_history.history['val_accuracy']
        loss += fine_tune_history.history['loss']
        val_loss += fine_tune_history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

plot_history(history, history_fine)


# === CONFUSION MATRIX ===
class_names = sorted(os.listdir(os.path.join(DATA_DIR, "train")))

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(10, 7))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Validation Set")
plt.show()


# === SAMPLE PREDICTIONS ===
sample_images, sample_labels = next(iter(val_ds))
preds = model.predict(sample_images)

plt.figure(figsize=(12, 12))
for i in range(9):
    idx = random.randint(0, sample_images.shape[0] - 1)
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[idx].numpy().astype("uint8"))
    true_label = class_names[np.argmax(sample_labels[idx].numpy())]
    pred_label = class_names[np.argmax(preds[idx])]
    color = "green" if true_label == pred_label else "red"
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
    plt.axis("off")
plt.show()
