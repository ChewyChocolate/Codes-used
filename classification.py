import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, callbacks

from google.colab import drive
drive.mount('/content/drive')

# === CONFIGURATION ===
DATA_DIR = "/content/drive/MyDrive/capsule_dataset_noBG"
IMG_SIZE = (224, 224)   # EfficientNetB0 default input size
BATCH_SIZE = 32
EPOCHS = 10
FINE_TUNE_EPOCHS = 5
MODEL_DIR = "/content/drive/MyDrive/capsule_model"
MODEL_NAME = "capsule_efficientnet_model"

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

# === PREFETCHING, SHUFFLING & NORMALIZATION ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

# === DATA AUGMENTATION (helps prevent overfitting) ===
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# === BUILD MODEL ===
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)

base_model.trainable = False  # freeze base for transfer learning

num_classes = train_ds.element_spec[1].shape[1]

model = models.Sequential([
    data_augmentation,
    normalization_layer,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(256, activation="relu"),   # fixed to a stable size instead of train_ds.cardinality()
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

# === COMPILE MODEL ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# === CALLBACKS ===
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
checkpoint = callbacks.ModelCheckpoint(
    filepath=f"{MODEL_DIR}/{MODEL_NAME}.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
)

# === TRAIN MODEL ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# === UNFREEZE & FINE-TUNE ===
base_model.trainable = True
# Fine-tune only top layers (avoid catastrophic forgetting)
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# === EXPORT TO TFLITE ===
# Load best model before exporting
best_model = tf.keras.models.load_model(f"{MODEL_DIR}/{MODEL_NAME}.keras")

tf.saved_model.save(best_model, f"{MODEL_DIR}/{MODEL_NAME}_saved")
converter = tf.lite.TFLiteConverter.from_saved_model(f"{MODEL_DIR}/{MODEL_NAME}_saved")
tflite_model = converter.convert()

with open(f"{MODEL_DIR}/{MODEL_NAME}.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Training complete. Model exported to TFLite.")
