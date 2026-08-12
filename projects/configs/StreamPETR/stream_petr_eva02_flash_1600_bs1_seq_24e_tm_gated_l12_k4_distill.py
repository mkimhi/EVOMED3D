_base_ = ['./stream_petr_eva02_flash_1600_bs1_seq_24e.py']

work_dir = './work_dirs/stream_petr_eva02_flash_1600_bs1_seq_24e_tm_gated_l12_k4_distill'
load_from = './work_dirs/stream_petr_eva02_flash_1600_bs1_seq_24e/latest.pth'
find_unused_parameters = False

data = dict(train=dict(queue_length=4))

model = dict(
    temporal_memory=dict(
        enabled=True,
        detector_adapter='streampetr',
        keyframe_interval=4,
        max_frames_between_full=4,
        updater='gated_residual',
        updater_cfg=dict(
            channels=256,
            feature_format='spatial',
            residual_scale_init=0.0,
        ),
        partial_layer='block12',
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
