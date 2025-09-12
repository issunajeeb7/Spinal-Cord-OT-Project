# HematomaDetectionYolov8/train.py

from ultralytics import YOLO
import torch
import os

# --- FIX STARTS HERE ---
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv 
import torch.nn as nn 

torch.serialization.add_safe_globals([
    DetectionModel, 
    nn.Sequential,
    Conv
])
# --- FIX ENDS HERE ---

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
   
if __name__ == "__main__":    
    model = YOLO("yolov8m.pt")

    params = {
        'batch': 16,
        'lr': 0.0437,
        'epochs': 80,
    }

    print("MODEL INFO: ")
    print(model.info())
    print('#################')
    
    # --- IMPORTANT CHANGE HERE ---
    # Provide the full, absolute path to your config.yaml file.
    config_file_path = '/content/drive/MyDrive/OT/Code/hematoma-localization-and-spinal-cord-segmentation/HematomaDetectionYolov8/config.yaml'
    
    # Check if the file exists before trying to train
    if not os.path.exists(config_file_path):
        print(f"ERROR: The config file was not found at {config_file_path}")
        print("Please make sure the path is correct and the file is in your Colab session.")
    else:
        print(f"Using config file at: {config_file_path}")
        model.train(
            data=config_file_path,  # Use the full path variable here
            epochs=params['epochs'],
            device='0',
            batch = params['batch'],
            lr0 = params['lr'],
            imgsz=(320,320),
            save=True,
            pretrained='yolov8m.pt',
            val=True,
        )
        metrics = model.val()
        metrics