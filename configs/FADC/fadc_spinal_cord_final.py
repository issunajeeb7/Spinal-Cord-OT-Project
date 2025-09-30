# configs/FADC/fadc_spinal_cord_final.py

_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/spinal_cord.py',  # <-- Using your new dataset config
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# You have 10 classes (0 through 9)
num_classes = 10

model = dict(
    decode_head=dict(
        num_classes=num_classes),
    auxiliary_head=dict(
        num_classes=num_classes),
    test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

# You can adjust the optimizer and other training parameters here if needed
# For example, let's set a smaller learning rate for fine-tuning
optimizer = dict(lr=0.001)