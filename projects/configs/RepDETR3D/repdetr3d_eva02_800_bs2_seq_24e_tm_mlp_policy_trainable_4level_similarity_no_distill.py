_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_similarity_no_distill'
load_from = './ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth'
find_unused_parameters = False

data = dict(train=dict(queue_length=5))

_online_policy_features = [
    'frame_number', 'info_frame_idx', 'is_scene_start', 'dt_s',
    'ego_x_m', 'ego_y_m', 'ego_z_m', 'ego_yaw_deg',
    'delta_ego_translation_m', 'ego_speed_mps', 'ego_accel_mps2',
    'yaw_rate_deg_s', 'abs_yaw_rate_deg_s', 'yaw_accel_deg_s2', 'abs_yaw_accel_deg_s2',
]

_online_policy_mean = [
    19.567702,
    19.567702,
    0.024921083,
    0.48656002,
    1182.1858,
    1385.3081,
    0.0,
    31.97956,
    2.5787889,
    5.1699474,
    0.27622647,
    -0.098026916,
    2.1476363,
    -0.004204151,
    1.6220828,
]

_online_policy_scale = [
    11.587536,
    11.587536,
    0.15588465,
    0.082544361,
    562.12445,
    585.10477,
    1.0,
    104.69116,
    1.79063,
    3.5718243,
    2.0322777,
    5.0668897,
    4.5902765,
    3.6931669,
    3.3178829,
]

model = dict(
    temporal_memory=dict(
        enabled=True,
        detector_adapter='repdetr3d',
        keyframe_interval=5,
        policy=dict(
            type='mlp',
            checkpoint=None,
            load_weights=False,
            feature_names=_online_policy_features,
            mean=_online_policy_mean,
            scale=_online_policy_scale,
            hidden_dim=32,
            labels=('easy', 'medium', 'hard', 'very_hard'),
            actions=dict(easy='L8', medium='L12', hard='L16', very_hard='L24'),
            fallback_action='L24',
        ),
        policy_supervision=dict(
            enabled=True,
            target='similarity',
            warmup_ratio=0.1,
            max_train_steps=42192,
            thresholds=(0.9, 0.76, 0.675),
            class_weights=(1.0, 1.0, 1.0, 3.0),
            weight=1.0,
        ),
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
