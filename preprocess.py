import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.distance import cdist

# --- Configuration ---

# ============================ THIS IS THE FIX ============================
# This path is now RELATIVE. It means "look for a folder called
# 'segmentation_dataset' inside the current directory".
# This will work correctly since your data is inside the FADC folder.
# =========================================================================
original_dataset_root = 'segmentation_dataset/segmentation dataset'
new_dataset_root = 'fadc_spinal_cord_dataset'
# =========================================================================


# This is the definitive palette from the paper's source code
COLOR_TO_ID = {
    (128, 0, 128): 0,    # Dorsal Space
    (170, 0, 0): 1,      # Dorsal Dura
    (85, 255, 0): 2,     # CSF
    (0, 85, 0): 3,       # Pia
    (0, 0, 170): 4,      # Spinal Cord
    (85, 85, 0): 5,      # Ventral Space
    (0, 170, 170): 6,    # Hematoma
    (255, 85, 0): 7,     # Dura/Pia complex
    (170, 170, 0): 8,    # Dura/Ventral complex
    (255, 0, 0): 9,      # Ventral Dura
    (0, 0, 0): 255,      # Background -> Mapped to ignore_index
}

# --- Conversion & File Handling Logic (No changes needed below) ---

palette_colors = np.array(list(COLOR_TO_ID.keys()))
palette_ids = np.array(list(COLOR_TO_ID.values()))

def convert_mask_final(mask_path):
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
    print("🚀 Starting FINAL dataset preprocessing with the correct path...")
    if not os.path.isdir(original_root):
        print(f"❌ FATAL ERROR: The source dataset directory was not found at:")
        print(f"   '{os.path.abspath(original_root)}'")
        print("   Please make sure the 'segmentation_dataset' folder is inside your FADC project folder.")
        return

    if os.path.isdir(new_root):
        shutil.rmtree(new_root)

    splits = {
        'training': ('train_images', 'train_masks'),
        'validation': ('val_images', 'val_masks'),
        'testing': ('test_images', 'test_masks')
    }
    
    for split, (img_f, mask_f) in splits.items():
        original_img_dir = os.path.join(original_root, img_f)
        original_mask_dir = os.path.join(original_root, mask_f)
        
        if not os.path.isdir(original_img_dir):
            print(f"\n⚠️ WARNING: Source folder for '{split}' images not found, skipping: {original_img_dir}")
            continue

        new_img_dir = os.path.join(new_root, 'images', split)
        new_ann_dir = os.path.join(new_root, 'annotations', split)
        os.makedirs(new_img_dir, exist_ok=True)
        os.makedirs(new_ann_dir, exist_ok=True)

        print(f"\n--- Processing '{split}' split ---")
        
        print(f"Copying images from '{original_img_dir}'...")
        for fname in tqdm(os.listdir(original_img_dir), desc=f"Copying {split} images"):
            shutil.copy(os.path.join(original_img_dir, fname), os.path.join(new_img_dir, fname))

        if not os.path.isdir(original_mask_dir):
            print(f"⚠️ WARNING: Source folder for '{split}' masks not found, skipping conversion: {original_mask_dir}")
            continue
            
        print(f"Converting masks from '{original_mask_dir}'...")
        for fname in tqdm(os.listdir(original_mask_dir), desc=f"Converting {split} masks"):
            mask = convert_mask_final(os.path.join(original_mask_dir, fname))
            if mask: mask.save(os.path.join(new_ann_dir, fname))

    print(f"\n✅ Preprocessing complete! Your dataset is ready at: '{new_root}'")

if __name__ == '__main__':
    process_and_restructure(original_dataset_root, new_dataset_root)