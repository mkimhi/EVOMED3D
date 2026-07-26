from projects.mmdet3d_plugin.models.temporal_memory.policies import build_frame_policy


def test_previous_l8_drift_policy_maps_similarity_to_actions():
    policy = build_frame_policy(dict(
        type='previous_l8_drift',
        reset_interval=4,
        thresholds=(0.8, 0.65, 0.55),
    ))

    assert policy.decide(1, True, policy_features={'previous_l8_similarity': 0.81}) == 'L8'
    assert policy.decide(1, True, policy_features={'previous_l8_similarity': 0.70}) == 'L12'
    assert policy.decide(1, True, policy_features={'previous_l8_similarity': 0.60}) == 'L16'
    assert policy.decide(1, True, policy_features={'previous_l8_similarity': 0.40}) == 'L24'


def test_previous_l8_drift_policy_resets_every_interval():
    policy = build_frame_policy(dict(type='previous_l8_drift', reset_interval=4))

    assert policy.decide(0, False, policy_features={'previous_l8_similarity': 0.95}) == 'L24'
    assert policy.decide(4, True, policy_features={'previous_l8_similarity': 0.95}) == 'L24'
