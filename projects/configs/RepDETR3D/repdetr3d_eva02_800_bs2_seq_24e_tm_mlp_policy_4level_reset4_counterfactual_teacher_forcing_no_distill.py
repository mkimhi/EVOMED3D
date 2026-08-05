_base_ = [
    './repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_'
    'reset5_similarity_online_offline_teacher_forcing_no_distill.py'
]

policy_checkpoint = 'outputs/counterfactual_policy_train/mlp_policy_counterfactual.pt'
target_csv = 'outputs/counterfactual_policy_train/counterfactual_policy_targets.csv'

work_dir = (
    './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_4level_'
    'reset4_counterfactual_teacher_forcing_no_distill'
)

model = dict(
    temporal_memory=dict(
        max_frames_between_full=4,
        policy=dict(
            checkpoint=policy_checkpoint,
            load_weights=True,
        ),
        policy_supervision=dict(
            _delete_=True,
            enabled=True,
            target='counterfactual_action',
            target_csv=target_csv,
            start_train_step=4219,
            max_train_steps=42192,
            residual_noise_multiplier=1.0,
            class_weights=(1.0, 1.0, 1.0, 3.0),
            weight=1.0,
        ),
        policy_offline_features='teacher_forcing',
        policy_offline_csv=(
            'outputs/policy_signal_stats/train/policy_signal_stats.csv'),
    ),
)

# The updater trains alone for the first 4219 steps. Afterward the loaded policy
# receives its counterfactual CE loss at one tenth of the updater learning rate.
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'temporal_memory_policy': dict(lr_mult=0.1),
        },
    ),
)
