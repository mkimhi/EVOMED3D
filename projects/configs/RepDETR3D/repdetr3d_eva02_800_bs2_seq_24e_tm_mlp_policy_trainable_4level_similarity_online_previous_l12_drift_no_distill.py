_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_similarity_no_distill.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_similarity_online_previous_l12_drift_no_distill'

_l12_policy_features = ['frame_number',
 'info_frame_idx',
 'is_scene_start',
 'dt_s',
 'ego_x_m',
 'ego_y_m',
 'ego_z_m',
 'ego_yaw_deg',
 'delta_ego_translation_m',
 'ego_speed_mps',
 'ego_accel_mps2',
 'yaw_rate_deg_s',
 'abs_yaw_rate_deg_s',
 'yaw_accel_deg_s2',
 'abs_yaw_accel_deg_s2',
 'previous_l12_similarity']
_l12_policy_mean = [19.567702,
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
 0.0]
_l12_policy_scale = [11.587536,
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
 1.0]

model = dict(
    temporal_memory=dict(
        policy=dict(
            feature_names=_l12_policy_features,
            mean=_l12_policy_mean,
            scale=_l12_policy_scale,
            requires_previous_l12_similarity=True,
        ),
    )
)
