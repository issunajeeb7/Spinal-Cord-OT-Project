# configs/_base_/datasets/spinal_cord.py

dataset_type = 'CustomDataset'
# This path points to the dataset you just created.
data_root = '/kaggle/input/fadc-spinal-cord-dataset/fadc_spinal_cord_dataset' 

# This is the definitive list of classes and colors from the paper's source code.
# This will prevent NaN errors.
CLASSES = (
    'Dorsal Space',         # 0
    'Dorsal Dura',          # 1
    'CSF',                  # 2
    'Pia',                  # 3
    'Spinal Cord',          # 4
    'Ventral Space',        # 5
    'Hematoma',             # 6
    'Dura/Pia complex',     # 7
    'Dura/Ventral complex', # 8
    'Ventral Dura'          # 9
)

PALETTE = [
    [128, 0, 128], [170, 0, 0], [85, 255, 0], [0, 85, 0], [0, 0, 170],
    [85, 85, 0], [0, 170, 170], [255, 85, 0], [170, 170, 0], [255, 0, 0]
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
    samples_per_gpu=1, # Set to 1 to prevent out-of-memory errors on your laptop
    workers_per_gpu=2,
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