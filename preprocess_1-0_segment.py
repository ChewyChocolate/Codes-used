import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import random

# === CONFIG ===
RAW_IMAGES_DIR = "dataset/segmentation_dataset/images"   # input images (already resized)
RAW_MASKS_DIR = "dataset/segmentation_dataset/masks"     # binary masks (same filenames as images)
OUTPUT_DIR = "capsule_segmentation_dataset"  # output dataset root
TRAIN_SPLIT = 0.8                   # 80% train, 20% val
AUG_PER_IMAGE = 5                   # augmentations per image

def augment_pair(img, mask, max_augmentations=3):
    """Apply up to max_augmentations random transformations to an image/mask pair."""
    augmentations = [
        "rotate", "hflip", "vflip", "brightness", "contrast",
        "translate", "scale", "noise", "blur"
    ]
    num_augs = random.randint(1, max_augmentations)
    choices = random.sample(augmentations, num_augs)
    
    for choice in choices:
        if choice == "rotate":
            angle = random.randint(-25, 25)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
            mask = mask.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)
        
        elif choice == "hflip":
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        
        elif choice == "vflip":
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        
        elif choice == "brightness":
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.6, 1.4))
        
        elif choice == "contrast":
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.6, 1.4))
        
        elif choice == "translate":
            dx = int(random.uniform(-0.1, 0.1) * img.width)
            dy = int(random.uniform(-0.1, 0.1) * img.height)
            img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=(0, 0, 0))
            mask = mask.transform(mask.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)
        
        elif choice == "scale":
            scale = random.uniform(0.8, 1.2)
            new_size = (int(img.width * scale), int(img.height * scale))
            img_resized = img.resize(new_size, Image.BICUBIC)
            mask_resized = mask.resize(new_size, Image.NEAREST)
            canvas_img = Image.new("RGB", img.size, (0, 0, 0))
            canvas_mask = Image.new("L", mask.size, 0)
            x_offset = (canvas_img.width - img_resized.width) // 2
            y_offset = (canvas_img.height - img_resized.height) // 2
            canvas_img.paste(img_resized, (x_offset, y_offset))
            canvas_mask.paste(mask_resized, (x_offset, y_offset))
            img, mask = canvas_img, canvas_mask
        
        elif choice == "noise":
            arr = np.array(img).astype("int16")
            noise = np.random.normal(0, 15, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype("uint8")
            img = Image.fromarray(arr)
        
        elif choice == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    return img, mask


# === ORGANIZE AND AUGMENT ===
def prepare_dataset():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"))
        os.makedirs(os.path.join(OUTPUT_DIR, split, "masks"))

    # Pair images and masks
    images = [f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_SPLIT)
    train_imgs, val_imgs = images[:split_idx], images[split_idx:]

    def process(images_list, split, augment=False):
        for idx, img_file in enumerate(images_list):
            img_path = os.path.join(RAW_IMAGES_DIR, img_file)
            mask_path = os.path.join(RAW_MASKS_DIR, img_file)

            if not os.path.exists(mask_path):
                print(f"⚠️ No mask found for {img_file}, skipping.")
                continue

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # Save original
            base_name = f"{idx:04d}.png"
            img.save(os.path.join(OUTPUT_DIR, split, "images", base_name))
            mask.save(os.path.join(OUTPUT_DIR, split, "masks", base_name))

            # Augment only training
            if augment:
                for i in range(AUG_PER_IMAGE):
                    aug_img, aug_mask = augment_pair(img, mask)
                    aug_name = f"{idx:04d}_aug{i}.png"
                    aug_img.save(os.path.join(OUTPUT_DIR, split, "images", aug_name))
                    aug_mask.save(os.path.join(OUTPUT_DIR, split, "masks", aug_name))

    process(train_imgs, "train", augment=True)
    process(val_imgs, "val", augment=False)

    print("✅ Binary segmentation dataset prepared at", OUTPUT_DIR)


if __name__ == "__main__":
    prepare_dataset()
