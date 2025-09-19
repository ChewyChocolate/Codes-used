import os
import re
import io
from rembg import remove, new_session
from PIL import Image, ImageOps

# === CONFIG ===
INPUT_DIR = "/home/Earl/Downloads/segmentation_dataset/segmentation_dataset_1"                 # folder with raw capsule images
OUTPUT_IMAGES_DIR = "dataset/segmentation_dataset/images"  # resized originals
OUTPUT_MASKS_DIR = "dataset/segmentation_dataset/masks"    # resized masks

TARGET_SIZE = (224, 224)
THRESHOLD = 128  # for binarization

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def generate_dataset():
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort(key=natural_sort_key)

    session = new_session("u2net")

    for idx, filename in enumerate(files, start=1):
        src_path = os.path.join(INPUT_DIR, filename)

        # --- Load & normalize orientation ---
        img = Image.open(src_path).convert("RGB")
        img = ImageOps.exif_transpose(img)

        # --- Resize ---
        img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)

        new_name = f"{idx:04d}.png"

        # Save resized image
        save_img_path = os.path.join(OUTPUT_IMAGES_DIR, new_name)
        img_resized.save(save_img_path, format="PNG")

        # --- Mask generation ---
        buf = io.BytesIO()
        img_resized.save(buf, format="PNG")
        mask_bytes = remove(buf.getvalue(), only_mask=True, session=session)
        buf.close()

        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_img = mask_img.resize(TARGET_SIZE, Image.NEAREST)

        # Binarize mask
        mask_bin = mask_img.point(lambda p: 255 if p > THRESHOLD else 0)

        save_mask_path = os.path.join(OUTPUT_MASKS_DIR, new_name)
        mask_bin.save(save_mask_path, format="PNG")

        print(f"✅ Processed {filename} -> {new_name}")

    print("\n🎯 Dataset ready in:")
    print("   Images:", OUTPUT_IMAGES_DIR)
    print("   Masks :", OUTPUT_MASKS_DIR)


if __name__ == "__main__":
    generate_dataset()
