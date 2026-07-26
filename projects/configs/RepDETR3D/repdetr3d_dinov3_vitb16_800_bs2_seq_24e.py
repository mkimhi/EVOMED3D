_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_dinov3_vitb16_800_bs2_seq_24e'

dinov3_vitb16_sim_fpn = dict(
    scale_factors=[4, 2, 1, 0.5],
    in_channels=768,
    out_channels=256,
    out_indices=[2, 3, 4, 5],
)

model = dict(
    img_roi_head=None,
    img_backbone=dict(
        _delete_=True,
        type='DINOv3ViT',
        img_size=(320, 800),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        num_register_tokens=4,
        mlp_type='mlp',
        sim_fpn=dinov3_vitb16_sim_fpn,
        pretrained='./ckpts/dinov3_vitb16_timm/pytorch_model.bin',
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
                name='repdetr3d_dinov3_vitb16_800_bs2_seq_24e',
                tags=['RepDETR3D', 'DINOv3', 'ViT-B', 'nuScenes'],
            ))
    ])

load_from = None
