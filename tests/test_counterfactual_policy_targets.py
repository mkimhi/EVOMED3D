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
