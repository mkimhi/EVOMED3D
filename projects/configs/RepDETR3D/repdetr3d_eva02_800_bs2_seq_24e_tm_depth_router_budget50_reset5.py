_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_depth_router_budget50_reset5'
load_from = './ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth'
find_unused_parameters = False

# Batch four reaches the 40 GB A100 limit after multiple routed-depth paths
# have been exercised.  Batch two is the safe training default on four GPUs.
# It still emits two independent per-sample router decisions per GPU.
# Evaluation is intentionally unbatched: one route is selected per temporal
# sample at deployment.
router_eval_cfg = dict(data=dict(samples_per_gpu=1))
data = dict(samples_per_gpu=2, train=dict(queue_length=5))

# The base config assumes eight GPUs.  Preserve its 24-epoch sample budget
# when running the documented four-GPU command.
num_iters_per_epoch = 28130 // (4 * 2)
evaluation = dict(interval=num_iters_per_epoch * 24)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)
runner = dict(type='IterBasedRunner', max_iters=num_iters_per_epoch * 24)

_online_router_features = (
    'frame_number', 'info_frame_idx', 'is_scene_start', 'dt_s',
    'ego_x_m', 'ego_y_m', 'ego_z_m', 'ego_yaw_deg',
    'delta_ego_translation_m', 'ego_speed_mps', 'ego_accel_mps2',
    'yaw_rate_deg_s', 'abs_yaw_rate_deg_s', 'yaw_accel_deg_s2',
    'abs_yaw_accel_deg_s2', 'frames_since_full',
)

model = dict(
    temporal_memory=dict(
        enabled=True,
        detector_adapter='repdetr3d',
        keyframe_interval=5,
        max_frames_between_full=5,
        partial_layers=(8, 12, 16),
        partial_layer='block8',
        updater='gated_residual',
        shared_memory_gate=True,
        updater_cfg=dict(channels=256, feature_format='spatial', residual_scale_init=0.0),
        projector_cfg=dict(
            in_channels=1024, out_channels=256, scale_factors=(4, 2, 1, 0.5),
            out_indices=(2, 3, 4, 5)),
        # This router is optimized by detection and compute/balance losses only.
        # It receives no annotation, GT, or current/previous drift feature.
        router=dict(
            type='depth_exit',
            actions=('L8', 'L12', 'L16', 'L24'),
            feature_names=_online_router_features,
            hidden_dim=128,
            temperature=1.0,
            costs=(8 / 24, 12 / 24, 16 / 24, 1.0),
            target_cost=0.5,
            budget_weight=1.0,
            balance_prior=(0.25, 0.25, 0.25, 0.25),
            balance_weight=0.01,
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
    tm_debug_compare_full_path=False,
    tm_assert_no_teacher_in_inference=True,
)
