import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

# === CONFIG ===
RAW_DATA_DIR = "dataset/to_upload"     # input folder with subfolders per brand
OUTPUT_DIR = "capsule_dataset"   # organized dataset
TRAIN_SPLIT = 0.8                # 80% train, 20% val
IMG_SIZE = (224, 224)            # resize target
AUG_PER_IMAGE = 5                # augmentations per image

# === AUGMENTATION FUNCTIONS ===
def random_crop_zoom(img, scale_range=(0.85, 1.0)):
    """Randomly crop and resize back to IMG_SIZE to simulate zoom."""
    w, h = img.size
    scale = random.uniform(*scale_range)
    new_w, new_h = int(w * scale), int(h * scale)

    if new_w < w and new_h < h:
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))

    return img.resize(IMG_SIZE)


def augment_image(img):
    aug_list = []

    for _ in range(AUG_PER_IMAGE):
        aug = img.copy()

        # Random rotation
        if random.random() < 0.7:
            aug = aug.rotate(random.randint(-25, 25), expand=True)

        # Random crop/zoom
        if random.random() < 0.7:
            aug = random_crop_zoom(aug, scale_range=(0.85, 1.0))

        # Random horizontal flip
        if random.random() < 0.5:
            aug = ImageOps.mirror(aug)

        # Random brightness
        if random.random() < 0.7:
            enhancer = ImageEnhance.Brightness(aug)
            aug = enhancer.enhance(random.uniform(0.7, 1.3))

        # Random contrast
        if random.random() < 0.7:
            enhancer = ImageEnhance.Contrast(aug)
            aug = enhancer.enhance(random.uniform(0.7, 1.3))

        # Random color/saturation
        if random.random() < 0.5:
            enhancer = ImageEnhance.Color(aug)
            aug = enhancer.enhance(random.uniform(0.8, 1.2))

        # Gaussian blur
        if random.random() < 0.3:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Add random noise
        if random.random() < 0.3:
            np_img = np.array(aug)
            noise = np.random.normal(0, 10, np_img.shape).astype(np.int16)
            np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            aug = Image.fromarray(np_img)

        # Resize back to target (safety)
        aug = aug.resize(IMG_SIZE)
        aug_list.append(aug)

    return aug_list


# === ORGANIZE AND AUGMENT ===
def prepare_dataset():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "train"))
    os.makedirs(os.path.join(OUTPUT_DIR, "val"))

    for brand in os.listdir(RAW_DATA_DIR):
        brand_path = os.path.join(RAW_DATA_DIR, brand)
        if not os.path.isdir(brand_path):
            continue

        os.makedirs(os.path.join(OUTPUT_DIR, "train", brand))
        os.makedirs(os.path.join(OUTPUT_DIR, "val", brand))

        images = [f for f in os.listdir(brand_path) if f.endswith(".png")]
        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_SPLIT)
        train_imgs, val_imgs = images[:split_idx], images[split_idx:]

        # Training images + augmentation
        for img_file in train_imgs:
            img_path = os.path.join(brand_path, img_file)
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)

            # Save original
            img.save(os.path.join(OUTPUT_DIR, "train", brand, img_file))

            # Save augmented
            for i, aug_img in enumerate(augment_image(img)):
                aug_name = f"{os.path.splitext(img_file)[0]}_aug{i}.png"
                aug_img.save(os.path.join(OUTPUT_DIR, "train", brand, aug_name))

        # Validation images (no aug)
        for img_file in val_imgs:
            img_path = os.path.join(brand_path, img_file)
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            img.save(os.path.join(OUTPUT_DIR, "val", brand, img_file))

    print("✅ Dataset prepared at", OUTPUT_DIR)


if __name__ == "__main__":
    prepare_dataset()
