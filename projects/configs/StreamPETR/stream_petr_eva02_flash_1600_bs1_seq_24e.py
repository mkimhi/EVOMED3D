_base_ = ['./stream_petr_eva02_flash_800_bs2_seq_24e.py']

num_gpus = 4
batch_size = 1
num_epochs = 24
num_iters_per_epoch = 28130 // (num_gpus * batch_size)

work_dir = './work_dirs/stream_petr_eva02_flash_1600_bs1_seq_24e'

ida_aug_conf = dict(
    resize_lim=(0.94, 1.25),
    final_dim=(640, 1600),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True,
)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='PETRFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'prev_exists'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels',
            'centers2d', 'depths', 'prev_exists', 'lidar2img', 'intrinsics',
            'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose',
            'ego_pose_inv'
        ],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
            'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
            'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=[
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv'
                ],
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'lidar2img', 'intrinsics', 'extrinsics',
                    'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv'
                ],
                meta_keys=(
                    'filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    train=dict(pipeline=train_pipeline, modality=input_modality),
    val=dict(pipeline=test_pipeline, modality=input_modality),
    test=dict(pipeline=test_pipeline, modality=input_modality),
)

evaluation = dict(interval=num_iters_per_epoch * num_epochs, pipeline=test_pipeline)

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=2)

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

load_from = './ckpts/eva02_L_coco_det_sys_o365_remapped.pth'
