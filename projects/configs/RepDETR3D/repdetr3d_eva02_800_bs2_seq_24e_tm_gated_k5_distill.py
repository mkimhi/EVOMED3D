_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k5_distill'
load_from = './ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth'

model = dict(
    temporal_memory=dict(
        enabled=True,
        detector_adapter='repdetr3d',
        keyframe_interval=5,
        updater='gated_residual',
        updater_cfg=dict(
            channels=256,
            feature_format='spatial',
            residual_scale_init=0.0,
        ),
        partial_layer='middle',
        memory_is_pre_pe=True,
        pe_mode='separate_recompute',
        freeze_pretrained_detector=True,
        use_ego_conditioning=False,
        use_distillation=True,
        distill_weight=1.0,
        train_with_teacher=True,
        inference_teacher_forbidden=True,
    ),
    tm_debug_shapes=False,
    tm_debug_latency=False,
    tm_debug_compare_full_path=False,
    tm_assert_no_teacher_in_inference=True,
)
