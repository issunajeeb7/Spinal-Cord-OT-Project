norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=10,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=10,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
dataset_type = 'CustomDataset'
data_root = 'C:/Users/issu/Documents/OT/fadc_spinal_cord_dataset'
CLASSES = ('Dorsal Space', 'Dura', 'CSF', 'Pia', 'Spinal Cord',
           'Ventral Space', 'Hematoma', 'Dura/Pia complex',
           'Dura/Ventral complex', 'Unknown Gray')
PALETTE = [[128, 0, 128], [128, 0, 0], [0, 128, 0], [85, 255, 0], [0, 0, 128],
           [128, 128, 0], [0, 128, 128], [255, 85, 0], [170, 170, 0],
           [128, 128, 128]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(690, 275), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_root='C:/Users/issu/Documents/OT/fadc_spinal_cord_dataset',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(690, 275), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        palette=[[128, 0, 128], [128, 0, 0], [0, 128, 0], [85, 255, 0],
                 [0, 0, 128], [128, 128, 0], [0, 128, 128], [255, 85, 0],
                 [170, 170, 0], [128, 128, 128]],
        classes=('Dorsal Space', 'Dura', 'CSF', 'Pia', 'Spinal Cord',
                 'Ventral Space', 'Hematoma', 'Dura/Pia complex',
                 'Dura/Ventral complex', 'Unknown Gray'),
        img_suffix='.png',
        seg_map_suffix='.png'),
    val=dict(
        type='CustomDataset',
        data_root='C:/Users/issu/Documents/OT/fadc_spinal_cord_dataset',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(690, 275),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        palette=[[128, 0, 128], [128, 0, 0], [0, 128, 0], [85, 255, 0],
                 [0, 0, 128], [128, 128, 0], [0, 128, 128], [255, 85, 0],
                 [170, 170, 0], [128, 128, 128]],
        classes=('Dorsal Space', 'Dura', 'CSF', 'Pia', 'Spinal Cord',
                 'Ventral Space', 'Hematoma', 'Dura/Pia complex',
                 'Dura/Ventral complex', 'Unknown Gray'),
        img_suffix='.png',
        seg_map_suffix='.png'),
    test=dict(
        type='CustomDataset',
        data_root='C:/Users/issu/Documents/OT/fadc_spinal_cord_dataset',
        img_dir='images/testing',
        ann_dir='annotations/testing',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(690, 275),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        palette=[[128, 0, 128], [128, 0, 0], [0, 128, 0], [85, 255, 0],
                 [0, 0, 128], [128, 128, 0], [0, 128, 128], [255, 85, 0],
                 [170, 170, 0], [128, 128, 128]],
        classes=('Dorsal Space', 'Dura', 'CSF', 'Pia', 'Spinal Cord',
                 'Ventral Space', 'Hematoma', 'Dura/Pia complex',
                 'Dura/Ventral complex', 'Unknown Gray'),
        img_suffix='.png',
        seg_map_suffix='.png'))
log_config = dict(
    interval=50, hooks=[dict(type='CustomizedTextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU')
num_classes = 10
work_dir = 'work_dirs/fadc_spinal_cord'
gpu_ids = [0]
auto_resume = False
