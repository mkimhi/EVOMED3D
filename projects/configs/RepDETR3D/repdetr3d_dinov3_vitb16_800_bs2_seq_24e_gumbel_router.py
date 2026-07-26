_base_ = ['./repdetr3d_dinov3_vitb16_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_dinov3_vitb16_800_bs2_seq_24e_gumbel_router'

model = dict(
    pts_bbox_head=dict(
        routing_mode='learned_gumbel_ambient',
        routing_keep_ratio=0.5,
        routing_keep_ratio_stopped=0.5,
        routing_keep_ratio_moderate=0.6,
        routing_keep_ratio_fast=0.7,
        routing_stopped_speed=0.5,
        routing_high_speed=12.0,
        routing_log_stats=False,
        routing_router=dict(
            type='GumbelTokenRouter',
            embed_dims=256,
            hidden_dims=256,
            keep_ratio=0.5,
            temperature=1.0,
            use_global_context=True,
        ),
    ),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='sepatch3d',
                name='repdetr3d_dinov3_vitb16_gumbel_router_800_bs2_seq_24e',
                tags=['RepDETR3D', 'DINOv3', 'ViT-B', 'GumbelRouter', 'nuScenes'],
            ))
    ])
