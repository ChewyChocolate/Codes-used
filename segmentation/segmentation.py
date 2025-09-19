# Install required packages
!pip install keras-unet-collection

# Mount Google Drive to access the dataset
from google.colab import drive
drive.mount('/content/drive')

import os
import tensorflow as tf
from tensorflow.keras import backend as K
from keras_unet_collection import models
import numpy as np
from glob import glob

# Dataset paths
base_dir = '/content/drive/MyDrive/capsule_segmentation_dataset'

train_img_dir = os.path.join(base_dir, 'train/images')
train_mask_dir = os.path.join(base_dir, 'train/masks')
val_img_dir = os.path.join(base_dir, 'val/images')
val_mask_dir = os.path.join(base_dir, 'val/masks')

# Get sorted lists of files
train_img_paths = sorted(glob(os.path.join(train_img_dir, '*.png')))
train_mask_paths = sorted(glob(os.path.join(train_mask_dir, '*.png')))
val_img_paths = sorted(glob(os.path.join(val_img_dir, '*.png')))
val_mask_paths = sorted(glob(os.path.join(val_mask_dir, '*.png')))

print(f"Train images: {len(train_img_paths)}, Val images: {len(val_img_paths)}")

# Image size
IMG_SIZE = 224

# Load and preprocess function
def load_and_preprocess(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))  # Already 224x224, but ensures consistency
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = tf.cast(mask, tf.float32) / 255.0  # Binary: 0 or 1

    return img, mask

# Create tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=1000).batch(8).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_img_paths, val_mask_paths))
val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(8).prefetch(tf.data.AUTOTUNE)

# Define U-Net model
model = models.unet_2d(
    input_size=(IMG_SIZE, IMG_SIZE, 3),
    filter_num=[64, 128, 256, 512, 1024],  # Standard U-Net filter progression
    n_labels=1,
    activation='ReLU',
    output_activation='Sigmoid',
    batch_norm=True,
    stack_num_down=2,  # 2 conv layers per downsampling block
    stack_num_up=2,    # 2 conv layers per upsampling block
    name='unet'
)

# Binary cross-entropy loss (no deep supervision needed for U-Net)
loss = tf.keras.losses.BinaryCrossentropy()

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

def iou(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + K.epsilon())

ious = []
for imgs, masks in val_ds:
    preds = model.predict(imgs)  # Single output
    for i in range(len(masks)):
        ious.append(iou(masks[i:i+1], preds[i:i+1]))

print(f"Mean IoU: {np.mean(ious)}")

# Load best model
model.load_weights('best_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('capsule_segmentation.tflite', 'wb') as f:
    f.write(tflite_model)

# Download
from google.colab import files
files.download('capsule_segmentation.tflite')

interpreter = tf.lite.Interpreter(model_path='capsule_segmentation.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Sample image
sample_img, _ = next(iter(val_ds.take(1)))
sample_img = sample_img[0:1]  # Batch of 1, already 224x224
interpreter.set_tensor(input_details[0]['index'], sample_img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print("TFLite output shape:", output.shape)  # Should be (1, 224, 224, 1)
