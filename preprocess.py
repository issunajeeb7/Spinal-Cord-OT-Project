import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.distance import cdist


# --- Configuration ---

original_dataset_root = '/content/segmentation dataset'
new_dataset_root = '/content/fadc_spinal_cord_dataset'


# ============================ ACTION REQUIRED ============================
# This dictionary is now complete, based on the colors YOU discovered.
# I have assigned class IDs based on the likely anatomical part.
# =========================================================================
COLOR_TO_ID = {
    # Class 0: Dorsal Space
    (128, 0, 128): 0,

    # Class 1: Dura (all shades of red)
    (64, 0, 0): 1,
    (128, 0, 0): 1,
    (192, 0, 0): 1,

    # Class 2: CSF (all shades of green)
    (0, 128, 0): 2,
    (1, 128, 0): 2,
    (2, 128, 0): 2,
    (3, 128, 0): 2,
    (8, 128, 0): 2,
    (28, 128, 0): 2,
    (42, 128, 0): 2,

    # Class 4: Spinal Cord
    (0, 0, 128): 4,

    # Class 5: Ventral Space
    (128, 128, 0): 5,

    # Class 6: Hematoma
    (0, 128, 128): 6,

    # Class 9: Unknown Gray (assigning a new class ID)
    (128, 128, 128): 9,

    # Ignore Index 255: Background
    (0, 0, 0): 255,
}
# =========================================================================


# --- Final Conversion Logic (No need to edit below) ---
palette_colors = np.array(list(COLOR_TO_ID.keys()))
palette_ids = np.array(list(COLOR_TO_ID.values()))


def convert_mask_final(mask_path):
    """Converts an RGB mask to a single-channel index mask by finding the closest color."""
    try:
        rgb_mask = Image.open(mask_path).convert('RGB')
        pixel_data = np.array(rgb_mask).reshape(-1, 3)
        distances = cdist(pixel_data, palette_colors)
        closest_indices = np.argmin(distances, axis=1)
        index_data = palette_ids[closest_indices]
        return Image.fromarray(index_data.reshape(rgb_mask.height, rgb_mask.width).astype(np.uint8))
    except Exception as e:
        print(f"Error converting {mask_path}: {e}")
        return None


def process_and_restructure(original_root, new_root):
    print("🚀 Starting FINAL dataset preprocessing with your complete custom palette...")
    if os.path.isdir(new_root):
        shutil.rmtree(new_root)

    splits = {
        'training': ('train_images', 'train_masks'),
        'validation': ('val_images', 'val_masks'),
        'testing': ('test_images', 'test_masks'),
    }

    for split, (img_f, mask_f) in splits.items():
        new_img_dir = os.path.join(new_root, 'images', split)
        new_ann_dir = os.path.join(new_root, 'annotations', split)
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_ann_dir, exist_ok=True)

        print(f"\n--- Processing '{split}' split ---")
        if os.path.isdir(os.path.join(original_root, img_f)):
            for fname in tqdm(os.listdir(os.path.join(original_root, img_f)), desc=f"Copying {split} images"):
                shutil.copy(os.path.join(original_root, img_f, fname), os.path.join(new_img_dir, fname))
        if os.path.isdir(os.path.join(original_root, mask_f)):
            for fname in tqdm(os.listdir(os.path.join(original_root, mask_f)), desc=f"Converting {split} masks"):
                mask = convert_mask_final(os.path.join(original_root, mask_f, fname))
                if mask:
                    mask.save(os.path.join(new_ann_dir, fname))

    print(f"\n✅ Preprocessing complete! Your dataset is ready at: '{new_root}'")


    # --- Final Verification ---
    print("\n🔍 Verifying a sample processed mask...")
    try:
        sample_mask_path = os.path.join(
            new_root, 'annotations/training',
            os.listdir(os.path.join(new_root, 'annotations/training'))[0]
        )
        mask_data = np.array(Image.open(sample_mask_path))
        unique_vals = np.unique(mask_data)
        print(f"Verification successful! Found class indices in sample mask: {unique_vals}")
        if len(unique_vals) > 2:
            print("This looks correct. You can now proceed to training.")
        else:
            print("Warning: Still seeing few classes. Double-check your COLOR_TO_ID mapping if issues persist.")
    except Exception as e:
        print(f"Could not verify mask. Error: {e}")


if __name__ == '__main__':
    process_and_restructure(original_dataset_root, new_dataset_root)
