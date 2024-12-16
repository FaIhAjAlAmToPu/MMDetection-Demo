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
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],  # Smaller anchors for small objects
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,  # Single class for detection
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
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

optim_wrapper = dict(
    optimizer=optimizer
)
