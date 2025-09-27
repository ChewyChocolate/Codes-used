import os
from pathlib import Path
from rembg import remove
from PIL import Image

# For timing and visualization
import time
import matplotlib.pyplot as plt

def remove_background_in_folder(folder_path, output_suffix="_nobg"):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ Folder does not exist: {folder}")
        return

    # Loop through all images in subfolders
    processed = []
    failed = []
    skipped = []
    start_time = time.time()
    for img_path in folder.rglob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            continue  # skip non-images

        print(f"\n---\nProcessing: {img_path.resolve()}")
        try:
            with Image.open(img_path) as img:
                # Resize image to 300x300 for faster background removal
                img_resized = img.resize((224, 224), Image.LANCZOS)

                # Remove background
                result = remove(img_resized)



                # Always save as PNG (preserve transparency if present)
                output_path = img_path.with_stem(img_path.stem + output_suffix).with_suffix(".png")
                try:
                    result.save(output_path)
                    print(f"✅ Saved: {output_path.resolve()}")
                except Exception as e:
                    print(f"❌ Failed to save {output_path.resolve()}: {e}")
                    print(f"⚠️ Skipping deletion of original: {img_path.resolve()}")
                    failed.append(str(img_path.resolve()))
                    continue  # Skip deletion if save failed

                # Confirm file exists before deleting original
                if not output_path.exists():
                    print(f"❌ Output file does not exist after save: {output_path.resolve()}")
                    print(f"⚠️ Skipping deletion of original: {img_path.resolve()}")
                    failed.append(str(img_path.resolve()))
                    continue

                # Remove the original image only if save succeeded
                try:
                    os.remove(img_path)
                    print(f"🗑️ Removed original: {img_path.resolve()}")
                    processed.append(f"{img_path.resolve()} → {output_path.resolve()}")
                except Exception as e:
                    print(f"⚠️ Could not remove {img_path.resolve()}: {e}")
                    skipped.append(str(img_path.resolve()))

        except Exception as e:
            print(f"⚠️ Failed: {img_path.resolve()} ({e})")
            failed.append(str(img_path.resolve()))

    print("\n=== SUMMARY ===")
    print(f"Processed and replaced: {len(processed)}")
    for p in processed:
        print(f"  {p}")
    print(f"Failed: {len(failed)}")
    for f in failed:
        print(f"  {f}")
    print(f"Skipped (could not remove original): {len(skipped)}")
    for s in skipped:
        print(f"  {s}")

    # Timing and visualization
    elapsed = time.time() - start_time
    print(f"⏱️ Total time elapsed: {elapsed:.2f} seconds")
    plt.figure(figsize=(6, 2))
    plt.bar(["Duration"], [elapsed], color="skyblue")
    plt.ylabel("Seconds")
    plt.title("Script Duration")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 🔹 Change this to your target folder
    target_folder = "/home/Earl/Downloads/Crab/KAGANG-LAND CRAB-20250927T125649Z-1-001/KAGANG-LAND CRAB"
    remove_background_in_folder(target_folder)
