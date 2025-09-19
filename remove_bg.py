import os
from pathlib import Path
from rembg import remove
from PIL import Image

def remove_background_in_folder(folder_path, output_suffix="_nobg"):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ Folder does not exist: {folder}")
        return

    # Loop through all images in subfolders
    for img_path in folder.rglob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            continue  # skip non-images

        try:
            with Image.open(img_path) as img:
                # Remove background
                result = remove(img)

                # Save output in the same folder with suffix
                output_path = img_path.with_stem(img_path.stem + output_suffix).with_suffix(".png")
                result.save(output_path)

                print(f"✅ Processed: {img_path} → {output_path}")

        except Exception as e:
            print(f"⚠️ Failed: {img_path} ({e})")


if __name__ == "__main__":
    # 🔹 Change this to your target folder
    target_folder = "/path/to/your/folder"
    remove_background_in_folder(target_folder)
