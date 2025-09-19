import os
from PIL import Image

# --- CONFIG ---
resize_width = 800   # target width
resize_height = 600  # target height
extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
# --------------

def resize_image(input_path, output_path, size):
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img_resized = img.resize(size, Image.LANCZOS)
            img_resized.save(output_path)
            print(f"Resized: {input_path}")
    except Exception as e:
        print(f"❌ Failed to process {input_path}: {e}")

def process_folder(root_folder):
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(subdir, file)
                resize_image(file_path, file_path, (resize_width, resize_height))

if __name__ == "__main__":
    # 🔹 Change this path to the folder you want to process
    #target_folder = r"C:\Users\YourName\Pictures\MyImages"  # Windows example
    target_folder = "/home/Earl/Downloads/segmentation_dataset/segmentation_dataset_1"    # Linux/Mac example
    
    if os.path.isdir(target_folder):
        process_folder(target_folder)
        print("✅ Done resizing all images in subfolders.")
    else:
        print(f"❌ Folder not found: {target_folder}")
