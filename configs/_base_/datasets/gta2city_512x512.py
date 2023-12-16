_base_ = './gta2city.py'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(957, 526), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # dict(type='PackSegInputsTargets'),
    dict(type='KeyMapper',
         mapping={
            'img':'target_img',
            'img_path':'target_path',
            'img_shape':'target_shape',
            'ori_shape':'target_ori_shape',
            'pad_shape':'target_pad_shape',
            'seg_map_path':'target_seg_map_path',
            'gt_bboxes':'target_gt_bboxes',
            'gt_seg_map':'target_gt_seg_map',
            'gt_keypoints':'target_gt_keypoints',
            'scale':'target_scale',
            'scale_factor':'target_scale_factor',
            'keep_ratio':'target_keep_ratio',
            'seg_fields':'target_seg_fields',
            'inputs':'targets',
            'data_samples':'target_data_samples'
            
         },
         auto_remap=True,
         transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', scale=(1024, 512), keep_ratio=True),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            # dict(type='PackSegInputsTargets')
         ]),
    dict(type='PackSegInputsTargets')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
