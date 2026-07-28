_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_frozen'
load_from = './ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth'
find_unused_parameters = False

# Evaluation is batch-one because deployment routes one temporal sample at once.
router_eval_cfg = dict(data=dict(samples_per_gpu=1))

model = dict(
    temporal_memory=dict(
        enabled=True,
        detector_adapter='repdetr3d',
        # MoD keeps the first eight and final block dense; reset-5 retains a
        # periodic full-memory refresh for temporal evaluation.
        keyframe_interval=5,
        max_frames_between_full=5,
        mod_router=dict(
            type='token_depth',
            routed_blocks=(9, 12, 15, 18, 21),
            capacity_ratio=0.5,
            router_hidden_channels=256,
            adapter_hidden_channels=256,
            train_backbone=False,
            adapter_warmup_only=False,
        ),
        memory_is_pre_pe=True,
        pe_mode='separate_recompute',
        freeze_backbone_only=True,
        use_distillation=False,
        train_with_teacher=False,
        inference_teacher_forbidden=True,
    ),
    tm_debug_shapes=False,
    tm_debug_latency=False,
    tm_assert_no_teacher_in_inference=True,
)
