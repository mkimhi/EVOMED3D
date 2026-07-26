_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_temporal_memory'
load_from = './ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth'

model = dict(
    temporal_memory=dict(
        enabled=False,
        detector_adapter='repdetr3d',
        keyframe_interval=3,
        updater='reuse',
        partial_layer='middle',
        partial_layer_options=dict(
            quarter='25%',
            middle='50%',
            three_quarter='75%',
        ),
        memory_is_pre_pe=True,
        pe_mode='separate_recompute',
        use_ego_conditioning=False,
        use_distillation=False,
        distill_weight=0.0,
        train_with_teacher=False,
        inference_teacher_forbidden=True,
    ),
    tm_debug_shapes=False,
    tm_debug_latency=False,
    tm_debug_compare_full_path=False,
    tm_assert_no_teacher_in_inference=True,
)
