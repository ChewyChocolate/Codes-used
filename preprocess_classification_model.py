import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
from tqdm import tqdm   # progress bar

# For timing and visualization
import time
import matplotlib.pyplot as plt

# === CONFIG ===
RAW_DATA_DIR = "/home/Earl/Downloads/Crab/temp1"   # change to your input folder
OUTPUT_DIR = "/home/Earl/Downloads/Crab/temp1_augmented"      # change to your output folder
TRAIN_SPLIT = 0.8                # 80% train, 20% val
IMG_SIZE = (224, 224)            # resize target
AUG_PER_IMAGE = 3                # augmentations per image

# === SUPPORTED IMAGE EXTENSIONS ===
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

# === AUGMENTATION FUNCTIONS ===
def random_crop_zoom(img, scale_range=(0.85, 1.0), target_size=(300, 300)):
    try:
        w, h = img.size
        scale = random.uniform(*scale_range)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        if new_w < w and new_h < h:
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h))

        return img.resize(target_size)
    except Exception as e:
        print(f"⚠️ Crop/zoom failed: {e}")
        return img.resize(target_size)

def augment_image(img, num_aug=AUG_PER_IMAGE, target_size=IMG_SIZE):
    aug_list = []
    for _ in range(num_aug):
        aug = img.copy()

        try:
            if random.random() < 0.7:
                aug = aug.rotate(random.randint(-25, 25), expand=True, fillcolor=(255, 255, 255))

            if random.random() < 0.7:
                aug = random_crop_zoom(aug, target_size=target_size)

            if random.random() < 0.5:
                aug = ImageOps.mirror(aug)

            if random.random() < 0.7:
                aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.7, 1.3))

            if random.random() < 0.7:
                aug = ImageEnhance.Contrast(aug).enhance(random.uniform(0.7, 1.3))

            if random.random() < 0.5:
                aug = ImageEnhance.Color(aug).enhance(random.uniform(0.8, 1.2))

            if random.random() < 0.3:
                aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            if random.random() < 0.3:
                np_img = np.array(aug)
                noise = np.random.normal(0, 10, np_img.shape).astype(np.int16)
                np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
                aug = Image.fromarray(np_img)

            aug = aug.resize(target_size)
        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}")
            aug = img.resize(target_size)

        aug_list.append(aug)
    return aug_list

# === ORGANIZE AND AUGMENT ===
def prepare_dataset():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "train"))
    os.makedirs(os.path.join(OUTPUT_DIR, "val"))

    start_time = time.time()

    for brand in os.listdir(RAW_DATA_DIR):
        brand_path = os.path.join(RAW_DATA_DIR, brand)
        if not os.path.isdir(brand_path):
            continue

        os.makedirs(os.path.join(OUTPUT_DIR, "train", brand))
        os.makedirs(os.path.join(OUTPUT_DIR, "val", brand))

        images = [f for f in os.listdir(brand_path) if f.lower().endswith(VALID_EXT)]
        if not images:
            print(f"⚠️ No images found in {brand_path}, skipping...")
            continue

        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_SPLIT)
        train_imgs, val_imgs = images[:split_idx], images[split_idx:]

        # Training images
        for img_file in tqdm(train_imgs, desc=f"Training {brand}", unit="img"):
            img_path = os.path.join(brand_path, img_file)
            try:
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            except Exception as e:
                print(f"⚠️ Could not open {img_path}: {e}")
                continue

            base_name, _ = os.path.splitext(img_file)
            img.save(os.path.join(OUTPUT_DIR, "train", brand, f"{base_name}.png"))

            for i, aug_img in enumerate(augment_image(img)):
                aug_name = f"{base_name}_aug_{i}.png"
                aug_img.save(os.path.join(OUTPUT_DIR, "train", brand, aug_name))

        # Validation images
        for img_file in tqdm(val_imgs, desc=f"Validation {brand}", unit="img"):
            img_path = os.path.join(brand_path, img_file)
            try:
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                base_name, _ = os.path.splitext(img_file)
                img.save(os.path.join(OUTPUT_DIR, "val", brand, f"{base_name}.png"))
            except Exception as e:
                print(f"⚠️ Could not open {img_path}: {e}")

    print("✅ Dataset prepared at", OUTPUT_DIR)

    # Timing and visualization
    elapsed = time.time() - start_time
    print(f"⏱️ Total time elapsed: {elapsed:.2f} seconds")
    plt.figure(figsize=(6, 2))
    plt.bar(["Duration"], [elapsed], color="skyblue")
    plt.ylabel("Seconds")
    plt.title("Script Duration")
    plt.tight_layout()
    plt.show()

# === RUN ===
if __name__ == "__main__":
    prepare_dataset()