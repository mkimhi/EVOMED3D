_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_frozen.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_frozen_adapter_warmup'
router_eval_cfg = dict(data=dict(samples_per_gpu=1))

model = dict(
    temporal_memory=dict(
        mod_router=dict(
            type='token_depth',
            routed_blocks=(9, 12, 15, 18, 21),
            capacity_ratio=0.5,
            router_hidden_channels=256,
            adapter_hidden_channels=256,
            train_backbone=False,
            adapter_warmup_only=True,
            warmup_iters=4000,
        ),
        # The warm-up aligns frozen-backbone routed features before detector loss.
        freeze_pretrained_detector=True,
    ),
)
