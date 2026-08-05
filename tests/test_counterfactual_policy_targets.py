import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / 'tools' / 'counterfactual_policy_targets.py'
SPEC = importlib.util.spec_from_file_location(
    'counterfactual_policy_targets', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MetricConfig = MODULE.MetricConfig
choose_oracle_action = MODULE.choose_oracle_action
matched_prediction_error = MODULE.matched_prediction_error


def box(class_name, x=0.0, score=1.0):
    return dict(
        class_name=class_name,
        translation=(float(x), 0.0, 0.0),
        size=(2.0, 4.0, 1.5),
        yaw=0.0,
        velocity=(0.0, 0.0),
        score=float(score),
    )


def test_matched_error_is_class_aware_and_penalizes_unmatched():
    config = MetricConfig()
    car = box('car', x=0.0)
    pedestrian = box('pedestrian', x=20.0)
    target = box('car', x=1.0)

    with_false_positive = matched_prediction_error(
        [car, pedestrian], [target], config, normalize=False)
    matched_only = matched_prediction_error(
        [car], [target], config, normalize=False)

    assert with_false_positive == pytest.approx(
        matched_only + config.false_positive_penalty)


def test_center_gate_leaves_far_same_class_boxes_unmatched():
    config = MetricConfig(center_gate_m=2.0)
    error = matched_prediction_error(
        [box('car', x=0.0)], [box('car', x=3.0)], config,
        normalize=False)
    assert error == pytest.approx(
        config.false_positive_penalty + config.false_negative_penalty)


def test_choose_shallowest_eligible_action():
    errors = {'L8': 1.04, 'L12': 1.02, 'L16': 1.01, 'L24': 1.00}
    assert choose_oracle_action(
        errors, max_matched_degradation=0.03) == 'L12'


def test_negative_degradation_keeps_shallow_action_eligible():
    errors = {'L8': 0.90, 'L12': 0.95, 'L16': 0.98, 'L24': 1.00}
    assert choose_oracle_action(
        errors, max_matched_degradation=0.0) == 'L8'


def test_optional_loss_constraint_can_reject_shallow_action():
    errors = {'L8': 1.01, 'L12': 1.01, 'L16': 1.01, 'L24': 1.00}
    losses = {'L8': 1.20, 'L12': 1.04, 'L16': 1.01, 'L24': 1.00}
    assert choose_oracle_action(
        errors,
        max_matched_degradation=0.05,
        losses=losses,
        max_loss_degradation=0.05,
    ) == 'L12'


def detection(score=0.9, x=1.0):
    return dict(
        detection_name='car',
        detection_score=score,
        translation=[x, 0.0, 0.0],
        size=[2.0, 4.0, 1.5],
        rotation=[1.0, 0.0, 0.0, 0.0],
        velocity=[0.0, 0.0],
    )


def test_previous_prediction_features_follow_canonical_action_and_scene_reset():
    frames = [
        {'token': 'a0', 'scene_token': 'A', 'canonical_action': 'L8'},
        {'token': 'a1', 'scene_token': 'A', 'canonical_action': 'L24'},
        {'token': 'b0', 'scene_token': 'B', 'canonical_action': 'L12'},
    ]
    branches = {
        action: {token: [] for token in ('a0', 'a1', 'b0')}
        for action in ('L8', 'L12', 'L16', 'L24')
    }
    branches['L8']['a0'] = [detection()]

    rows = MODULE.build_previous_prediction_rows(frames, branches)

    assert rows[0]['num_annotations'] == 0.0
    assert rows[1]['num_annotations'] == 1.0
    assert rows[2]['num_annotations'] == 0.0


def test_prediction_features_apply_score_threshold():
    features = MODULE.prediction_features([
        detection(score=0.09), detection(score=0.8, x=3.0)
    ])
    assert features['num_annotations'] == 1.0
    assert features['avg_object_distance_m'] == pytest.approx(3.0)


def test_manifest_requires_shared_pre_frame_state():
    with pytest.raises(ValueError, match='shared pre-frame state'):
        MODULE.validate_manifest({'shared_pre_frame_state': False})


def test_build_target_rows_records_errors_degradation_and_oracle_action():
    frames = [
        {'token': 'a0', 'scene_token': 'A', 'canonical_action': 'L24'},
    ]
    branches = {
        'L8': {'a0': [box('car', x=3.0)]},
        'L12': {'a0': [box('car', x=1.0)]},
        'L16': {'a0': [box('car', x=0.5)]},
        'L24': {'a0': [box('car', x=0.0)]},
    }
    rows = MODULE.build_target_rows(
        frames=frames,
        branch_results=branches,
        ground_truth={'a0': [box('car', x=0.0)]},
        metadata={'a0': {'token': 'a0', 'frame_number': 0.0}},
        metric_config=MetricConfig(),
        max_matched_degradation=0.1,
    )
    assert rows[0]['oracle_action'] == 'L12'
    assert rows[0]['error_L24'] == pytest.approx(0.0)
    assert rows[0]['degradation_L16'] == pytest.approx(rows[0]['error_L16'])
