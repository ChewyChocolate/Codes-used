import os
import random
from PIL import Image, ImageEnhance

# === CONFIG ===
OBJECTS_DIR = "/home/Earl/Downloads/Crab/crab_raw_dataset_nobg"       # folder with class subfolders (PNG with transparency)
BACKGROUNDS_DIR = "/home/Earl/Downloads/Random BG" # folder with random background images
OUTPUT_DIR = "/home/Earl/Downloads/Crab/temp1"     # where results are saved
N = 3                           # how many background variations per object
IMG_SIZE = (224, 224)           # resize final output

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect background file paths
background_files = [os.path.join(BACKGROUNDS_DIR, f) 
                    for f in os.listdir(BACKGROUNDS_DIR) 
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not background_files:
    raise ValueError("No background images found in BACKGROUNDS_DIR.")

print(f"Found {len(background_files)} backgrounds.")

# === Helper: Apply random augmentations ===
def augment_object(obj_img: Image.Image, bg_size=IMG_SIZE) -> Image.Image:
    # Compute max scale so object fits inside background
    max_scale = min(bg_size[0] / obj_img.width, bg_size[1] / obj_img.height, 1.2)
    min_scale = min(0.3, max_scale)
    scale = random.uniform(min_scale, max_scale)
    new_size = (max(1, int(obj_img.width * scale)), max(1, int(obj_img.height * scale)))
    obj_img = obj_img.resize(new_size, Image.LANCZOS)

    # Random rotation (with transparent fill)
    angle = random.uniform(-30, 30)
    obj_img = obj_img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    # If after rotation, object is too big, resize again
    if obj_img.width > bg_size[0] or obj_img.height > bg_size[1]:
        scale2 = min(bg_size[0] / obj_img.width, bg_size[1] / obj_img.height, 1.0)
        new_size2 = (max(1, int(obj_img.width * scale2)), max(1, int(obj_img.height * scale2)))
        obj_img = obj_img.resize(new_size2, Image.LANCZOS)

    # Random flip
    if random.random() < 0.5:
        obj_img = obj_img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        obj_img = obj_img.transpose(Image.FLIP_TOP_BOTTOM)

    # Random brightness / contrast
    if random.random() < 0.7:
        enhancer = ImageEnhance.Brightness(obj_img)
        obj_img = enhancer.enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.7:
        enhancer = ImageEnhance.Contrast(obj_img)
        obj_img = enhancer.enhance(random.uniform(0.7, 1.3))

    return obj_img

# === MAIN LOOP ===

# --- Process subfolders as classes ---
for class_name in os.listdir(OBJECTS_DIR):
    class_dir = os.path.join(OBJECTS_DIR, class_name)
    if os.path.isdir(class_dir):
        # Make output class folder
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        # Get object images in this class
        object_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[{class_name}] Found {len(object_files)} objects.")

        for obj_path in object_files:
            obj_name = os.path.splitext(os.path.basename(obj_path))[0]
            base_obj = Image.open(obj_path).convert("RGBA")

            for i in range(N):
                # Pick random background
                background_path = random.choice(background_files)
                bg = Image.open(background_path).convert("RGB")
                bg = bg.resize(IMG_SIZE)

                # Apply augmentation to object (pass background size)
                obj_aug = augment_object(base_obj, bg_size=IMG_SIZE)

                # Random position (inside background)
                max_x = max(0, bg.width - obj_aug.width)
                max_y = max(0, bg.height - obj_aug.height)
                pos = (random.randint(0, max_x), random.randint(0, max_y))

                # Paste object
                bg.paste(obj_aug, pos, obj_aug)

                # Ensure output is RGB (JPEG does not support alpha)
                out_img = bg.convert("RGB")

                # Save output as JPG
                out_path = os.path.join(output_class_dir, f"{obj_name}_mix{i+1}.jpg")
                out_img.save(out_path, format="JPEG", quality=95)

# --- Process image files directly in OBJECTS_DIR ---
object_files = [os.path.join(OBJECTS_DIR, f) for f in os.listdir(OBJECTS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(OBJECTS_DIR, f))]
if object_files:
    print(f"[ROOT] Found {len(object_files)} objects.")
    output_class_dir = os.path.join(OUTPUT_DIR, "root")
    os.makedirs(output_class_dir, exist_ok=True)
    for obj_path in object_files:
        obj_name = os.path.splitext(os.path.basename(obj_path))[0]
        base_obj = Image.open(obj_path).convert("RGBA")
        for i in range(N):
            background_path = random.choice(background_files)
            bg = Image.open(background_path).convert("RGB")
            bg = bg.resize(IMG_SIZE)
            obj_aug = augment_object(base_obj, bg_size=IMG_SIZE)
            max_x = max(0, bg.width - obj_aug.width)
            max_y = max(0, bg.height - obj_aug.height)
            pos = (random.randint(0, max_x), random.randint(0, max_y))
            bg.paste(obj_aug, pos, obj_aug)
            out_img = bg.convert("RGB")
            out_path = os.path.join(output_class_dir, f"{obj_name}_mix{i+1}.jpg")
            out_img.save(out_path, format="JPEG", quality=95)

print(f"✅ Done! All augmented mixed images saved in {OUTPUT_DIR}")