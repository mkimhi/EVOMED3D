_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_no_distill.py']

num_gpus = 8
batch_size = 4
num_epochs = 24
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
total_iters = num_epochs * num_iters_per_epoch

data = dict(samples_per_gpu=batch_size)
evaluation = dict(interval=total_iters)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)
runner = dict(type='IterBasedRunner', max_iters=total_iters)

work_dir = './work_dirs/repdetr3d_eva02_800_bs4_seq_24e_tm_l8_drift_policy_reset4_no_distill'

model = dict(
    temporal_memory=dict(
        keyframe_interval=4,
        max_frames_between_full=4,
        policy=dict(
            _delete_=True,
            type='previous_l8_drift',
            reset_interval=4,
            thresholds=(0.8, 0.65, 0.55),
            fallback_action='L24',
        ),
        policy_supervision=dict(enabled=False),
        updater='gated_residual',
        shared_memory_gate=True,
        partial_layers=(8, 12, 16),
        updater_cfg=dict(channels=256, feature_format='spatial', residual_scale_init=0.0),
        partial_layer='block8',
        projector_cfg=dict(in_channels=1024, out_channels=256, scale_factors=(4, 2, 1, 0.5), out_indices=(2, 3, 4, 5)),
        memory_is_pre_pe=True,
        pe_mode='separate_recompute',
        freeze_pretrained_detector=True,
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
