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
PREDICTION_SCORE_FLOOR = 0.10
FULL_FEATURE_COSINE_AGGREGATION = 'mean_of_level_mean_channelwise_cosine'


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


def full_feature_cosine_statistics(previous: Sequence[torch.Tensor], current: Sequence[torch.Tensor]):
    """Return compact per-level sufficient statistics for the L24 cosine metric."""
    if not previous or not current or len(previous) != len(current):
        raise ValueError('previous and current must be non-empty lists of equal length')
    levels = []
    for level_index, (before, after) in enumerate(zip(previous, current)):
        if before.shape != after.shape or before.ndim <= 2:
            raise ValueError('feature tensors must have matching shapes with channel dimension 2')
        cosine = F.cosine_similarity(before.float(), after.float(), dim=2)
        levels.append({
            'level_index': level_index,
            'cosine_sum': float(cosine.sum().detach().cpu()),
            'vector_count': int(cosine.numel()),
        })
    return {'aggregation': FULL_FEATURE_COSINE_AGGREGATION, 'levels': levels}


def reconstruct_full_feature_cosine(statistics) -> float:
    """Reconstruct ``full_feature_cosine`` exactly from per-level summaries."""
    if not isinstance(statistics, dict) or statistics.get('aggregation') != FULL_FEATURE_COSINE_AGGREGATION:
        raise ValueError('invalid full-L24 cosine aggregation metadata')
    levels = statistics.get('levels')
    if not isinstance(levels, list) or not levels:
        raise ValueError('full-L24 cosine statistics require non-empty levels')
    means = []
    for level in levels:
        count = level.get('vector_count', 0)
        if not isinstance(count, int) or count < 1:
            raise ValueError('full-L24 cosine level has invalid vector_count')
        means.append(float(level['cosine_sum']) / count)
    return float(np.mean(means))


def full_feature_cosine(previous: Sequence[torch.Tensor], current: Sequence[torch.Tensor]) -> float:
    """Mean channel-wise cosine similarity across full L24 feature tensors."""
    return reconstruct_full_feature_cosine(full_feature_cosine_statistics(previous, current))


def filter_predictions_by_score(predictions, score_floor: float = PREDICTION_SCORE_FLOOR):
    """Keep only predictions meeting the pre-specified detection-score floor."""
    if not np.isfinite(score_floor) or score_floor < 0.0:
        raise ValueError('score_floor must be a finite non-negative value')
    filtered = []
    for prediction in predictions:
        score = prediction.get('detection_score', prediction.get('score'))
        if score is None:
            raise ValueError('prediction is missing detection_score/score')
        if float(score) >= score_floor:
            filtered.append(prediction)
    return filtered


def zero_free_data_limits(values, padding_fraction: float = 0.05):
    """Return finite padded limits without imposing a zero baseline."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('axis limits require at least one finite value')
    if not 0.0 <= padding_fraction < 1.0:
        raise ValueError('padding_fraction must be in [0, 1)')
    minimum, maximum = float(values.min()), float(values.max())
    span = maximum - minimum
    if span == 0.0:
        span = max(abs(minimum) * 0.1, 1e-6)
    lower, upper = minimum - padding_fraction * span, maximum + padding_fraction * span
    if minimum > 0.0:
        lower = max(np.nextafter(0.0, 1.0), minimum * (1.0 - padding_fraction))
    if maximum < 0.0:
        upper = min(np.nextafter(0.0, -1.0), maximum * (1.0 - padding_fraction))
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError('could not construct finite informative axis limits')
    return float(lower), float(upper)


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


def build_panel_summary(frame: pd.DataFrame, repetitions: int, seed: int):
    """Return quintile and correlation summaries for all four paper panels."""
    quantiles, correlations = [], []
    for offset, drift_column in (('dt1', 'drift_dt1'), ('dt2', 'drift_dt2')):
        for outcome in ('prediction_loss', 'box_loss'):
            summary, correlation = scene_bootstrap_summary(
                frame, drift_column, outcome, repetitions, seed)
            quantiles.append(summary.assign(offset=offset, outcome=outcome))
            correlations.append({'offset': offset, 'outcome': outcome, **correlation})
    return (pd.concat(quantiles, ignore_index=True),
            pd.DataFrame(correlations))


def validate_target_tables(selection: pd.DataFrame, drift: pd.DataFrame, predictions,
                           expected_scenes: int = 100, targets_per_scene: int = 20):
    """Validate exact collector token coverage before any ground-truth lookup."""
    expected_targets = expected_scenes * targets_per_scene
    required_selection = {'token', 'scene_token'}
    required_drift = {'token', 'scene_token', 'drift_dt1', 'drift_dt2'}
    missing = required_selection - set(selection.columns)
    if missing:
        raise ValueError('selection.csv missing columns: {}'.format(sorted(missing)))
    missing = required_drift - set(drift.columns)
    if missing:
        raise ValueError('full_l24_drift.csv missing columns: {}'.format(sorted(missing)))
    if len(selection) != expected_targets or selection['token'].nunique() != expected_targets:
        raise ValueError('collector selection must contain exactly {} unique target tokens'.format(
            expected_targets))
    scene_sizes = selection.groupby('scene_token').size()
    if (len(scene_sizes) != expected_scenes or
            not scene_sizes.eq(targets_per_scene).all()):
        raise ValueError('collector selection must contain exactly {} targets per scene'.format(
            targets_per_scene))
    if len(drift) != expected_targets or drift['token'].duplicated().any():
        raise ValueError('drift table must contain exactly {} unique target tokens'.format(
            expected_targets))
    if not isinstance(predictions, dict):
        raise ValueError('results_nusc.json must contain a results mapping')
    selection_tokens = set(selection['token'])
    if set(drift['token']) != selection_tokens:
        raise ValueError('drift token coverage does not exactly match selection.csv')
    if set(predictions) != selection_tokens:
        raise ValueError('prediction token coverage does not exactly match selection.csv')
    selection_scenes = selection.set_index('token')['scene_token']
    drift_scenes = drift.set_index('token')['scene_token'].reindex(selection_scenes.index)
    if not selection_scenes.equals(drift_scenes):
        raise ValueError('drift token-to-scene mapping does not exactly match selection.csv')
    if not np.isfinite(drift[['drift_dt1', 'drift_dt2']].to_numpy(float)).all():
        raise ValueError('drift table contains non-finite values')
    merged = selection.merge(
        drift[['token', 'scene_token', 'drift_dt1', 'drift_dt2']],
        on=['token', 'scene_token'], how='inner', validate='one_to_one')
    if len(merged) != expected_targets:
        raise ValueError('selection/drift join must retain exactly {} targets'.format(
            expected_targets))
    return merged


def validate_collection_manifest(manifest) -> Dict[str, Path]:
    """Validate Task 2 provenance and return its existing artifact paths."""
    if manifest.get('representation_source') != 'base_full_l24_backbone_neck':
        raise ValueError('manifest must use base full L24 backbone/neck features')
    if manifest.get('prediction_mode') != 'naive_reuse_k2' or manifest.get('keyframe_interval') != 2:
        raise ValueError('manifest must describe naive reuse K=2 predictions')
    required = {'representation_source', 'prediction_mode', 'keyframe_interval',
                'selection_path', 'drift_path', 'drift_statistics_path', 'prediction_path',
                'dataset_version', 'selected_scene_tokens', 'selected_target_tokens',
                'selection_sha256', 'qualified_scene_count', 'selected_target_count'}
    missing = required - set(manifest)
    if missing:
        raise ValueError('manifest missing keys: {}'.format(sorted(missing)))
    paths = {'selection': Path(manifest['selection_path']), 'drift': Path(manifest['drift_path']),
             'drift_statistics': Path(manifest['drift_statistics_path']),
             'predictions': Path(manifest['prediction_path'])}
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError('manifest artifact paths do not exist: {}'.format(missing_paths))
    if not isinstance(manifest['dataset_version'], str) or not manifest['dataset_version']:
        raise ValueError('manifest must record a non-empty nuScenes dataset_version')
    scenes, tokens = manifest['selected_scene_tokens'], manifest['selected_target_tokens']
    if not isinstance(scenes, list) or not scenes or not isinstance(tokens, list) or not tokens:
        raise ValueError('manifest must record selected scene and target token lists')
    if len(scenes) != manifest['qualified_scene_count'] or len(tokens) != manifest['selected_target_count']:
        raise ValueError('manifest selected token counts do not match recorded counts')
    selection_bytes = paths['selection'].read_bytes()
    import hashlib
    if hashlib.sha256(selection_bytes).hexdigest() != manifest['selection_sha256']:
        raise ValueError('selection.csv digest does not match manifest provenance')
    selection = pd.read_csv(paths['selection'])
    if {'token', 'scene_token'} - set(selection.columns):
        raise ValueError('selection.csv lacks token/scene_token provenance columns')
    if selection['token'].tolist() != tokens:
        raise ValueError('selection.csv selected target tokens do not match manifest')
    if list(dict.fromkeys(selection['scene_token'])) != scenes:
        raise ValueError('selection.csv selected scene tokens do not match manifest')
    return paths
