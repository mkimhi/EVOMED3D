_base_ = ['./stream_petr_vov_flash_800_bs2_seq_24e.py']

work_dir = './work_dirs/stream_petr_eva02_flash_800_bs2_seq_24e'

eva02_stride16_sim_fpn = dict(
    scale_factors=[1],
    in_channels=1024,
    out_channels=256,
    out_indices=[4],
)

model = dict(
    position_level=0,
    stride=16,
    img_neck=None,
    img_backbone=dict(
        _delete_=True,
        type='EVAViT',
        img_size=320,
        patch_size=16,
        window_size=16,
        global_window_size=20,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        window_block_indexes=(
            list(range(0, 2)) + list(range(3, 5)) +
            list(range(6, 8)) + list(range(9, 11)) +
            list(range(12, 14)) + list(range(15, 17)) +
            list(range(18, 20)) + list(range(21, 23))
        ),
        sim_fpn=eva02_stride16_sim_fpn,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cp=True,
        flash_attn=True,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(
    _delete_=True,
    type='OptimizerHook',
    grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[21096],
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
                name='stream_petr_eva02_flash_800_bs2_seq_24e',
                tags=['StreamPETR', 'EVA02-L', 'flash-attn', 'no-deformable-attn', 'nuScenes'],
            ))
    ])

load_from = './ckpts/eva02_L_coco_det_sys_o365_remapped.pth'
resume_from = None
