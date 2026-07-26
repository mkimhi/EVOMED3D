_base_ = ['./stream_petr_dinov3_vitb16_flash_704_bs2_seq_24e_full_tokens.py']

work_dir = './work_dirs/stream_petr_dinov3_vitb16_flash_704_bs4_seq_24e_full_tokens'

num_gpus = 8
batch_size = 4
num_epochs = 24
num_iters_per_epoch = 28130 // (num_gpus * batch_size)

data = dict(samples_per_gpu=batch_size)

find_unused_parameters = True

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(with_cp=False)))))

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
evaluation = dict(interval=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)

lr_config = dict(
    _delete_=True,
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[num_iters_per_epoch * num_epochs // 2],
    gamma=0.1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='sepatch3d',
                name='stream_petr_dinov3_vitb16_flash_704_bs4_seq_24e_full_tokens',
                tags=['StreamPETR', 'DINOv3', 'ViT-B', 'bs4', 'full-tokens', 'nuScenes'],
            ))
    ])
