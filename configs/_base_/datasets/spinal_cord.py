# configs/_base_/datasets/spinal_cord.py

dataset_type = 'CustomDataset'
# Use the absolute path to your dataset
data_root = 'C:/Users/issu/Documents/OT/fadc_spinal_cord_dataset'

# Define the classes and palette based on your discovered colors
# The order of CLASSES must match your class IDs (0, 1, 2...).
CLASSES = (
    'Dorsal Space',      # 0
    'Dura',              # 1
    'CSF',               # 2
    'Pia',               # 3 - Placeholder, as it was not in your final palette
    'Spinal Cord',       # 4
    'Ventral Space',     # 5
    'Hematoma',          # 6
    'Dura/Pia complex',  # 7 - Placeholder
    'Dura/Ventral complex',# 8 - Placeholder
    'Unknown Gray'       # 9
)

PALETTE = [
    [128, 0, 128],   # 0: Dorsal Space
    [128, 0, 0],     # 1: Dura (using the most common red)
    [0, 128, 0],     # 2: CSF (using the most common green)
    [85, 255, 0],    # 3: Pia (placeholder color from original spec)
    [0, 0, 128],     # 4: Spinal Cord
    [128, 128, 0],   # 5: Ventral Space
    [0, 128, 128],   # 6: Hematoma
    [255, 85, 0],    # 7: Dura/Pia complex (placeholder)
    [170, 170, 0],   # 8: Dura/Ventral complex (placeholder)
    [128, 128, 128]  # 9: Unknown Gray
]

# --- Standard MMSegmentation data pipeline ---

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(690, 275), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(690, 275),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4, # Lower this to 2 or 1 if you get memory errors
    workers_per_gpu=2, # For Windows, it's often better to use fewer workers
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline,
        palette=PALETTE,
        classes=CLASSES,
        img_suffix='.png',
        seg_map_suffix='.png'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline,
        palette=PALETTE,
        classes=CLASSES,
        img_suffix='.png',
        seg_map_suffix='.png'),        
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/testing',
        ann_dir='annotations/testing',
        pipeline=test_pipeline,
        palette=PALETTE,
        classes=CLASSES,
        img_suffix='.png',
        seg_map_suffix='.png'))        
        