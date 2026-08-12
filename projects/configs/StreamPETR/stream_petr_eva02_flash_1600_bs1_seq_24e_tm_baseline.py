_base_ = ['./stream_petr_eva02_flash_1600_bs1_seq_24e.py']

work_dir = './work_dirs/stream_petr_eva02_flash_1600_bs1_seq_24e_tm_baseline'

model = dict(
    temporal_memory=dict(
        enabled=False,
        detector_adapter='streampetr',
        keyframe_interval=1,
        updater='reuse',
        partial_layer='block12',
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
