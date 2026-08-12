_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_online_offline_teacher_forcing_no_distill.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_motion_reset_feature_noise_no_distill'

model = dict(
    temporal_memory=dict(
        max_frames_between_full=4,
        adaptive_reset=dict(
            enabled=True,
            max_frames_between_full=2,
            ego_speed_mps=2.0,
            max_object_speed_mps=5.0,
        ),
        policy_supervision=dict(
            feature_noise_std=0.10,
        ),
    )
)
