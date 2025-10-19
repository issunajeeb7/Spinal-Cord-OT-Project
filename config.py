import torch

# Set the device to CUDA if a compatible GPU is available, otherwise use the CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CORRECTED: The definitive list of classes for the project ---
CLASSES = [
    'background',
    'blood/saline',
    'dura',
    'pia',
    'csf',
    'spinal cord',
    'hematoma',
    'dura/pia complex',
    'extradural space',
    'dura/extradural space',
]

# --- ADDED: Corresponding color map for visualization ---
# This list matches the order of the CLASSES list.
LABEL_COLORS_LIST = [
    [0, 0, 0],         # background
    [128, 0, 128],     # blood/saline
    [128, 0, 0],       # dura
    [0, 128, 0],       # pia
    [128, 128, 0],     # csf
    [0, 0, 128],       # spinal cord
    [0, 128, 128],     # hematoma
    [64, 0, 0],        # dura/pia complex
    [128, 128, 128],   # extradural space
    [192, 0, 0]        # dura/extradural space
]

# Automatically determine the number of classes based on the length of the list.
NUM_CLASSES = len(CLASSES)