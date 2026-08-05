#!/usr/bin/env python
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


ACTIONS = ('L8', 'L12', 'L16', 'L24')


@dataclass(frozen=True)
class MetricConfig:
    center_gate_m: float = 4.0
    confidence_weight: float = 0.25
    center_weight: float = 1.0
    size_weight: float = 0.25
    yaw_weight: float = 0.25
    velocity_weight: float = 0.25
    false_positive_penalty: float = 1.0
    false_negative_penalty: float = 1.0
    box_only: bool = False


def _vector(item, name, length):
    value = np.asarray(item.get(name, (0.0,) * length), dtype=np.float64)
    return value[:length]


def _pair_cost(prediction, target, config):
    center_distance = float(np.linalg.norm(
        _vector(prediction, 'translation', 2)
        - _vector(target, 'translation', 2)))
    if center_distance > config.center_gate_m:
        return None
    target_size = np.maximum(np.abs(_vector(target, 'size', 3)), 1e-6)
    size_error = float(np.mean(np.abs(
        _vector(prediction, 'size', 3) - _vector(target, 'size', 3))
        / target_size))
    yaw_delta = float(prediction.get('yaw', 0.0)) - float(target.get('yaw', 0.0))
    yaw_error = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))) / math.pi
    velocity_error = float(np.linalg.norm(
        _vector(prediction, 'velocity', 2)
        - _vector(target, 'velocity', 2)))
    cost = (
        config.center_weight * center_distance / max(config.center_gate_m, 1e-6)
        + config.size_weight * size_error
        + config.yaw_weight * yaw_error
        + config.velocity_weight * velocity_error
    )
    if not config.box_only:
        cost += config.confidence_weight * (
            1.0 - float(prediction.get('score', 0.0)))
    return cost


def matched_prediction_error(predictions, targets, config=None, normalize=True):
    config = config or MetricConfig()
    predictions = list(predictions)
    targets = list(targets)
    matched_predictions = set()
    matched_targets = set()
    total = 0.0
    classes = sorted({
        item['class_name'] for item in predictions + targets
    })
    for class_name in classes:
        pred_ids = [
            index for index, item in enumerate(predictions)
            if item['class_name'] == class_name
        ]
        target_ids = [
            index for index, item in enumerate(targets)
            if item['class_name'] == class_name
        ]
        if not pred_ids or not target_ids:
            continue
        matrix = np.full((len(pred_ids), len(target_ids)), np.inf)
        for row, pred_id in enumerate(pred_ids):
            for column, target_id in enumerate(target_ids):
                cost = _pair_cost(
                    predictions[pred_id], targets[target_id], config)
                if cost is not None:
                    matrix[row, column] = cost
        finite = np.isfinite(matrix)
        if not finite.any():
            continue
        fill = (
            float(matrix[finite].max())
            + config.false_positive_penalty
            + config.false_negative_penalty
        )
        rows, columns = linear_sum_assignment(
            np.where(finite, matrix, fill))
        for row, column in zip(rows, columns):
            if not finite[row, column]:
                continue
            total += float(matrix[row, column])
            matched_predictions.add(pred_ids[row])
            matched_targets.add(target_ids[column])
    total += config.false_positive_penalty * (
        len(predictions) - len(matched_predictions))
    total += config.false_negative_penalty * (
        len(targets) - len(matched_targets))
    if normalize:
        total /= max(1, len(predictions) + len(targets))
    return float(total)


def choose_oracle_action(
    errors,
    max_matched_degradation,
    losses=None,
    max_loss_degradation=None,
):
    missing = set(ACTIONS) - set(errors)
    if missing:
        raise ValueError(f'missing action errors: {sorted(missing)}')
    if losses is not None:
        missing_losses = set(ACTIONS) - set(losses)
        if missing_losses:
            raise ValueError(f'missing action losses: {sorted(missing_losses)}')
        if max_loss_degradation is None:
            raise ValueError('max_loss_degradation is required with losses')
    for action in ACTIONS:
        matched_ok = (
            float(errors[action]) - float(errors['L24'])
            <= float(max_matched_degradation)
        )
        loss_ok = losses is None or (
            float(losses[action]) - float(losses['L24'])
            <= float(max_loss_degradation)
        )
        if matched_ok and loss_ok:
            return action
    return 'L24'
