_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k3_l8_distill'
load_from = './ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth'

data = dict(
    train=dict(queue_length=3),
)

model = dict(
    temporal_memory=dict(
        enabled=True,
        detector_adapter='repdetr3d',
        keyframe_interval=3,
        policy=dict(type='fixed_interval', partial_action='L8'),
        updater='gated_residual',
        shared_memory_gate=True,
        partial_layers=(8, 12, 16),
        updater_cfg=dict(
            channels=256,
            feature_format='spatial',
            residual_scale_init=0.0,
        ),
        partial_layer='block8',
        projector_cfg=dict(
            in_channels=1024,
            out_channels=256,
            scale_factors=(4, 2, 1, 0.5),
            out_indices=(2, 3, 4, 5),
        ),
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
