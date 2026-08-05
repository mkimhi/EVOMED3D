"""Pure helpers for the base-L24 naive-reuse risk analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


CENTER_GATE_M = 4.0
VELOCITY_SCALE_MPS = 10.0
WEIGHTS = {'confidence': 0.5, 'center': 0.2, 'size': 0.1, 'yaw': 0.1, 'velocity': 0.1}
UNMATCHED_PENALTY = 1.0
_INVALID_MATCH_COST = 1e6


@dataclass(frozen=True)
class _SceneGroup:
    scene_token: str
    rows: Tuple[dict, ...]

    def reuse_indices(self, keyframe_interval: int) -> List[int]:
        return [index for index in range(2, len(self.rows))
                if index % keyframe_interval != 0]

    def evenly_spaced_reuse_rows(self, count: int, keyframe_interval: int) -> List[dict]:
        indices = self.reuse_indices(keyframe_interval)
        selected = np.linspace(0, len(indices) - 1, count, dtype=int)
        return [{**self.rows[indices[offset]], 'frame_index': indices[offset]}
                for offset in selected]


def contiguous_scene_groups(data_infos: Iterable[dict]) -> List[_SceneGroup]:
    """Return dataset-order scene groups, rejecting ambiguous sequence metadata."""
    groups = []
    seen_tokens, closed_scenes = set(), set()
    current_scene, current_rows = None, []
    for info in data_infos:
        if not isinstance(info, dict) or not {'token', 'scene_token'} <= info.keys():
            raise ValueError('every data info must contain token and scene_token')
        token, scene = info['token'], info['scene_token']
        if token in seen_tokens:
            raise ValueError('duplicate token in data infos: {}'.format(token))
        seen_tokens.add(token)
        if current_scene is None:
            current_scene = scene
        elif scene != current_scene:
            groups.append(_SceneGroup(current_scene, tuple(current_rows)))
            closed_scenes.add(current_scene)
            if scene in closed_scenes:
                raise ValueError('scene appears in non-contiguous groups: {}'.format(scene))
            current_scene, current_rows = scene, []
        current_rows.append(info)
    if current_scene is not None:
        groups.append(_SceneGroup(current_scene, tuple(current_rows)))
    return groups


def select_reuse_targets(data_infos, num_scenes=100, targets_per_scene=20,
                         keyframe_interval=2) -> List[dict]:
    """Select deterministic, balanced non-keyframe targets with two-frame history."""
    if num_scenes < 1 or targets_per_scene < 1 or keyframe_interval < 2:
        raise ValueError('num_scenes, targets_per_scene, and keyframe_interval must be positive')
    eligible = [group for group in contiguous_scene_groups(data_infos)
                if len(group.reuse_indices(keyframe_interval)) >= targets_per_scene]
    if len(eligible) < num_scenes:
        raise ValueError('insufficient scenes with {} eligible K={} reuse frames'.format(
            targets_per_scene, keyframe_interval))
    return [row for group in eligible[:num_scenes]
            for row in group.evenly_spaced_reuse_rows(targets_per_scene, keyframe_interval)]


def full_feature_cosine(previous: Sequence[torch.Tensor], current: Sequence[torch.Tensor]) -> float:
    """Mean channel-wise cosine similarity across full L24 feature tensors."""
    if not previous or not current or len(previous) != len(current):
        raise ValueError('previous and current must be non-empty lists of equal length')
    values = []
    for before, after in zip(previous, current):
        if before.shape != after.shape or before.ndim <= 2:
            raise ValueError('feature tensors must have matching shapes with channel dimension 2')
        values.append(F.cosine_similarity(before.float(), after.float(), dim=2).mean())
    return float(torch.stack(values).mean().detach().cpu())


def _yaw(rotation):
    w, x, y, z = np.asarray(rotation, dtype=float)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pair_components(prediction: dict, target: dict) -> Dict[str, float]:
    prediction_center = np.asarray(prediction['translation'], dtype=float)
    target_center = np.asarray(target['translation'], dtype=float)
    center_m = np.linalg.norm(prediction_center[:2] - target_center[:2])
    prediction_size = np.asarray(prediction['size'], dtype=float)
    target_size = np.asarray(target['size'], dtype=float)
    velocity = (np.asarray(prediction['velocity'], dtype=float)[:2] -
                np.asarray(target['velocity'], dtype=float)[:2])
    yaw_delta = _yaw(prediction['rotation']) - _yaw(target['rotation'])
    return {
        'confidence': 1.0 - float(prediction['score']),
        'center': center_m / CENTER_GATE_M,
        'size': float(np.mean(np.abs(np.log(np.clip(prediction_size, 1e-6, None) /
                                             np.clip(target_size, 1e-6, None))))),
        'yaw': float(abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))) / np.pi),
        'velocity': float(np.linalg.norm(velocity) / VELOCITY_SCALE_MPS),
    }


def frame_error(predictions, targets, include_confidence: bool) -> float:
    """Class-aware, 4 m center-gated Hungarian error for one frame."""
    predictions, targets = list(predictions), list(targets)
    denominator = max(len(predictions), len(targets), 1)
    total, matched_predictions, matched_targets = 0.0, set(), set()
    keys = tuple(WEIGHTS) if include_confidence else tuple(key for key in WEIGHTS if key != 'confidence')
    for name in sorted({box['name'] for box in predictions + targets}):
        prediction_indices = [i for i, box in enumerate(predictions) if box['name'] == name]
        target_indices = [i for i, box in enumerate(targets) if box['name'] == name]
        if not prediction_indices or not target_indices:
            continue
        matrix = np.full((len(prediction_indices), len(target_indices)), _INVALID_MATCH_COST)
        for row, prediction_index in enumerate(prediction_indices):
            for column, target_index in enumerate(target_indices):
                components = _pair_components(predictions[prediction_index], targets[target_index])
                if components['center'] <= 1.0:
                    matrix[row, column] = sum(WEIGHTS[key] * components[key] for key in keys)
        rows, columns = linear_sum_assignment(matrix)
        for row, column in zip(rows, columns):
            if matrix[row, column] < _INVALID_MATCH_COST:
                total += matrix[row, column]
                matched_predictions.add(prediction_indices[row])
                matched_targets.add(target_indices[column])
    total += UNMATCHED_PENALTY * (len(predictions) - len(matched_predictions))
    total += UNMATCHED_PENALTY * (len(targets) - len(matched_targets))
    return total / denominator


def scene_bootstrap_summary(frame: pd.DataFrame, drift_column: str, outcome_column: str,
                            repetitions: int, seed: int):
    """Summarize five drift quintiles with complete-scene bootstrap CIs."""
    required = {'scene_token', drift_column, outcome_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError('frame missing columns: {}'.format(sorted(missing)))
    if frame['scene_token'].nunique() < 2:
        raise ValueError('analysis requires at least two scenes')
    if frame[drift_column].nunique() < 5:
        raise ValueError('too few distinct drift values for five quantiles')
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    frame = frame.copy()
    frame['_drift_quantile'] = pd.qcut(frame[drift_column], 5, labels=False, duplicates='raise') + 1
    scenes = frame['scene_token'].unique()
    quantiles = list(range(1, 6))
    samples = {quantile: [] for quantile in quantiles}
    correlations = []
    rng = np.random.default_rng(seed)
    scene_frames = {scene: group for scene, group in frame.groupby('scene_token', sort=False)}
    for _ in range(repetitions):
        selected = rng.choice(scenes, size=len(scenes), replace=True)
        resampled = pd.concat([scene_frames[scene] for scene in selected], ignore_index=True)
        for quantile in quantiles:
            samples[quantile].append(resampled.loc[
                resampled['_drift_quantile'] == quantile, outcome_column].mean())
        correlations.append(spearmanr(resampled[drift_column], resampled[outcome_column]).statistic)
    summary = pd.DataFrame([{
        'drift_quantile': quantile,
        'drift_mean': frame.loc[frame['_drift_quantile'] == quantile, drift_column].mean(),
        'outcome_mean': frame.loc[frame['_drift_quantile'] == quantile, outcome_column].mean(),
        'ci_low': np.nanpercentile(samples[quantile], 2.5),
        'ci_high': np.nanpercentile(samples[quantile], 97.5),
        'frames': int((frame['_drift_quantile'] == quantile).sum()),
    } for quantile in quantiles])
    correlation = {
        'spearman': float(spearmanr(frame[drift_column], frame[outcome_column]).statistic),
        'ci_low': float(np.nanpercentile(correlations, 2.5)),
        'ci_high': float(np.nanpercentile(correlations, 97.5)),
    }
    return summary, correlation


def validate_collection_manifest(manifest) -> Dict[str, Path]:
    """Validate Task 2 provenance and return its existing artifact paths."""
    required = {'representation_source', 'prediction_mode', 'keyframe_interval',
                'selection_path', 'drift_path', 'prediction_path'}
    missing = required - set(manifest)
    if missing:
        raise ValueError('manifest missing keys: {}'.format(sorted(missing)))
    if manifest['representation_source'] != 'base_full_l24_backbone_neck':
        raise ValueError('manifest must use base full L24 backbone/neck features')
    if manifest['prediction_mode'] != 'naive_reuse_k2' or manifest['keyframe_interval'] != 2:
        raise ValueError('manifest must describe naive reuse K=2 predictions')
    paths = {'selection': Path(manifest['selection_path']), 'drift': Path(manifest['drift_path']),
             'predictions': Path(manifest['prediction_path'])}
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError('manifest artifact paths do not exist: {}'.format(missing_paths))
    return paths
