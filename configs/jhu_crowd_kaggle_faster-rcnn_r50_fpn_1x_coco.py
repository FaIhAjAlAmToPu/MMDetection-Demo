# Dataset configuration
data_root = '/kaggle/working/JHU-CROWD++-2/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(500, 500), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(500, 500), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=data_root + 'train/_annotations.coco.json',
        data_prefix=dict(img=data_root + 'train/'),
        metainfo=dict(
            classes=('head',),
            palette=[(220, 20, 60)],
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file=data_root + 'valid/_annotations.coco.json',
        data_prefix=dict(img=data_root + 'valid/'),
        metainfo=dict(
            classes=('head',),
            palette=[(220, 20, 60)],
        ),
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
    dataset=dict(
        ann_file=data_root + 'test/_annotations.coco.json',
        data_prefix=dict(img=data_root + 'test/'),
        metainfo=dict(
            classes=('head',),
            palette=[(220, 20, 60)],
        ),
        pipeline=test_pipeline
    )
)

# Model and optimizer configuration
model = dict(
    type='FasterRCNN',
    backbone=dict(type='ResNet', depth=50, init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    rpn_head=dict(type='RPNHead'),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(num_classes=1)  # Number of classes in the dataset
    ),
    train_cfg=dict(
        rpn=dict(assigner=dict(type='MaxIoUAssigner', iou_calculator=dict(type='BboxOverlaps2D'))),
        rcnn=dict(assigner=dict(type='MaxIoUAssigner', iou_calculator=dict(type='BboxOverlaps2D')))
    )
)

optimizer = dict(
    type='Adam', 
    lr=0.0001,  # Adjusted learning rate
    weight_decay=0.0001  # Added weight decay to prevent overfitting
)

# Learning rate scheduler
lr_config = dict(
    policy='step',
    step=[30, 40],
    gamma=0.1
)

train_cfg = dict(
    max_epochs=50,
    grad_clip=dict(max_norm=10, norm_type=2)
)

val_evaluator = dict(
    type='CocoMetric', 
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox'
)

test_evaluator = dict(
    type='CocoMetric', 
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox'
)

# Runtime settings
log_level = 'INFO'
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3)  # Save checkpoint every 5 epochs
)

work_dir = '/kaggle/working/faster_rcnn_jhu_crowd'
