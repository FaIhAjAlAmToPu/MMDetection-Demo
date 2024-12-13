_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

param_scheduler = [
    dict(
        type='StepLR',
        step=[30, 40],  # Reduce learning rate at these epochs
        gamma=0.1)
]


data_root = '/kaggle/working/JHU-CROWD++-2/'

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=data_root + 'train/_annotations.coco.json',
        data_prefix=dict(img=data_root + 'train/')))

val_dataloader = dict(
    dataset=dict(
        ann_file=data_root + 'valid/_annotations.coco.json',
        data_prefix=dict(img=data_root + 'valid/')))

test_dataloader = dict(
    dataset=dict(
        ann_file=data_root + 'test/_annotations.coco.json',
        data_prefix=dict(img=data_root + 'test/')))

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1)))  # Update with your class count

default_hooks = dict(
    checkpoint=dict(interval=5),
)

work_dir = '/kaggle/working/faster_rcnn_jhu_crowd'

train_cfg = dict(max_epochs=50)
