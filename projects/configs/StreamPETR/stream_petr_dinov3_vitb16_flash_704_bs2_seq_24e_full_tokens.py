_base_ = ['./stream_petr_r50_flash_704_bs2_seq_24e.py']

work_dir = './work_dirs/stream_petr_dinov3_vitb16_flash_704_bs2_seq_24e_full_tokens'

dinov3_vitb16_sim_fpn = dict(
    scale_factors=[4, 2, 1, 0.5],
    in_channels=768,
    out_channels=256,
    out_indices=[2, 3, 4, 5],
)

model = dict(
    # DINO's simple FPN returns [stride4, stride8, stride16, stride32].
    # StreamPETR R50 uses the stride16 map for PETR tokens, so select level 2.
    position_level=2,
    stride=16,
    img_roi_head=None,
    img_neck=None,
    img_backbone=dict(
        _delete_=True,
        type='DINOv3ViT',
        img_size=(256, 704),
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
    pts_bbox_head=dict(
        routing_mode='learned_gumbel_ambient',
        routing_keep_ratio=1.0,
        routing_ambient_context_len=1,
        routing_router=dict(
            type='GumbelTokenRouter',
            embed_dims=256,
            hidden_dims=256,
            keep_ratio=1.0,
            temperature=1.0,
            use_global_context=True,
        ),
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
                name='stream_petr_dinov3_vitb16_flash_704_bs2_seq_24e_full_tokens',
                tags=['StreamPETR', 'DINOv3', 'ViT-B', 'full-tokens', 'nuScenes'],
            ))
    ])

load_from = None
resume_from = None
