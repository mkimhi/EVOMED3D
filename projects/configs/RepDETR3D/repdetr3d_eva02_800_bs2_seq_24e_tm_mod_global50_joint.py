_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_frozen.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_joint'
router_eval_cfg = dict(data=dict(samples_per_gpu=1))

model = dict(
    temporal_memory=dict(
        mod_router=dict(
            type='token_depth',
            routed_blocks=(9, 12, 15, 18, 21),
            capacity_ratio=0.5,
            router_hidden_channels=256,
            adapter_hidden_channels=256,
            train_backbone=True,
            adapter_warmup_only=False,
            warmup_iters=4000,
        ),
        # N-vium-style joint phase: after router/adapter warm-up, optimize the
        # EVA backbone, detector, routers, and adapters together.
        freeze_backbone_only=False,
        freeze_pretrained_detector=False,
    ),
)
