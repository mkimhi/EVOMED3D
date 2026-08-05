#!/usr/bin/env python
import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ACTIONS = ('L8', 'L12', 'L16', 'L24')
ONLINE_FEATURES = (
    'frame_number', 'info_frame_idx', 'is_scene_start', 'dt_s',
    'ego_x_m', 'ego_y_m', 'ego_z_m', 'ego_yaw_deg',
    'delta_ego_translation_m', 'ego_speed_mps', 'ego_accel_mps2',
    'yaw_rate_deg_s', 'abs_yaw_rate_deg_s', 'yaw_accel_deg_s2',
    'abs_yaw_accel_deg_s2',
)
OFFLINE_FEATURES = (
    'num_annotations', 'avg_object_distance_m', 'median_object_distance_m',
    'min_object_distance_m', 'p10_object_distance_m',
    'p90_object_distance_m', 'near_object_count_10m',
    'near_object_count_20m', 'near_object_count_30m',
    'near_object_count_50m', 'avg_box_length_m', 'avg_box_width_m',
    'avg_box_height_m', 'avg_box_volume_m3', 'avg_box_area_m2',
    'avg_object_speed_mps', 'max_object_speed_mps',
    'camera_annotation_count', 'max_camera_annotation_count',
    'min_camera_annotation_count', 'avg_camera_annotation_count',
    'avg_2d_box_area_px', 'max_2d_box_area_px',
    'new_object_proxy_count', 'matched_object_proxy_count',
    'disappeared_object_proxy_count', 'new_object_proxy_fraction',
)


@dataclass(frozen=True)
class MetricConfig:
    center_gate_m: float = 4.0
    velocity_scale_mps: float = 10.0
    confidence_weight: float = 0.5
    center_weight: float = 0.2
    size_weight: float = 0.1
    yaw_weight: float = 0.1
    velocity_weight: float = 0.1
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
    pred_size = np.maximum(np.abs(_vector(prediction, 'size', 3)), 1e-6)
    target_size = np.maximum(np.abs(_vector(target, 'size', 3)), 1e-6)
    size_error = float(np.mean(np.abs(np.log(pred_size / target_size))))
    yaw_delta = float(prediction.get('yaw', 0.0)) - float(target.get('yaw', 0.0))
    yaw_error = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))) / math.pi
    velocity_error = float(np.linalg.norm(
        _vector(prediction, 'velocity', 2)
        - _vector(target, 'velocity', 2))) / max(config.velocity_scale_mps, 1e-6)
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
        total /= max(1, len(predictions), len(targets))
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


def zero_offline_features():
    return {name: 0.0 for name in OFFLINE_FEATURES}


def prediction_features(detections, score_threshold=0.1, ego_xy=None):
    detections = [
        item for item in detections
        if float(item.get('detection_score', 0.0)) >= score_threshold
    ]
    features = zero_offline_features()
    if not detections:
        return features
    translations = np.asarray(
        [item['translation'][:2] for item in detections], dtype=np.float64)
    if ego_xy is not None:
        translations = translations - np.asarray(ego_xy, dtype=np.float64)[:2]
    distances = np.linalg.norm(translations, axis=1)
    sizes = np.asarray(
        [item['size'][:3] for item in detections], dtype=np.float64)
    velocities = np.asarray(
        [item.get('velocity', (0.0, 0.0))[:2] for item in detections],
        dtype=np.float64,
    )
    speeds = np.linalg.norm(velocities, axis=1)
    features.update(
        num_annotations=float(len(detections)),
        avg_object_distance_m=float(distances.mean()),
        median_object_distance_m=float(np.median(distances)),
        min_object_distance_m=float(distances.min()),
        p10_object_distance_m=float(np.quantile(distances, 0.1)),
        p90_object_distance_m=float(np.quantile(distances, 0.9)),
        avg_box_length_m=float(sizes[:, 0].mean()),
        avg_box_width_m=float(sizes[:, 1].mean()),
        avg_box_height_m=float(sizes[:, 2].mean()),
        avg_box_volume_m3=float(np.prod(sizes, axis=1).mean()),
        avg_box_area_m2=float(np.prod(sizes[:, :2], axis=1).mean()),
        avg_object_speed_mps=float(speeds.mean()),
        max_object_speed_mps=float(speeds.max()),
    )
    for radius in (10, 20, 30, 50):
        features[f'near_object_count_{radius}m'] = float(
            (distances < radius).sum())
    return features


def build_previous_prediction_rows(frames, branch_results, metadata=None):
    rows = []
    previous_scene = None
    previous_features = None
    for frame in frames:
        scene = str(frame.get('scene_token', ''))
        if scene != previous_scene or previous_features is None:
            rows.append(zero_offline_features())
        else:
            rows.append(dict(previous_features))
        action = frame['canonical_action']
        token = str(frame['token'])
        frame_metadata = {} if metadata is None else metadata[token]
        ego_xy = None
        if 'ego_x_m' in frame_metadata and 'ego_y_m' in frame_metadata:
            ego_xy = (frame_metadata['ego_x_m'], frame_metadata['ego_y_m'])
        previous_features = prediction_features(
            branch_results[action][token], ego_xy=ego_xy)
        previous_scene = scene
    return rows


def validate_manifest(manifest):
    if manifest.get('shared_pre_frame_state') is not True:
        raise ValueError(
            'counterfactual manifest must declare shared pre-frame state')
    return manifest


def build_target_rows(
    frames,
    branch_results,
    ground_truth,
    metadata,
    metric_config=None,
    max_matched_degradation=0.05,
    loss_rows=None,
    max_loss_degradation=None,
):
    metric_config = metric_config or MetricConfig()
    previous_rows = build_previous_prediction_rows(
        frames, branch_results, metadata=metadata)
    output = []
    for frame, previous_features in zip(frames, previous_rows):
        token = str(frame['token'])
        if token not in metadata:
            raise KeyError(f'missing policy metadata for token {token}')
        if token not in ground_truth:
            raise KeyError(f'missing ground truth for token {token}')
        errors = {}
        for action in ACTIONS:
            if token not in branch_results[action]:
                raise KeyError(f'missing {action} prediction for token {token}')
            errors[action] = matched_prediction_error(
                branch_results[action][token],
                ground_truth[token],
                metric_config,
            )
        losses = None if loss_rows is None else loss_rows[token]
        row = dict(metadata[token])
        row.update(previous_features)
        row.update({f'error_{action}': errors[action] for action in ACTIONS})
        row.update({
            f'degradation_{action}': errors[action] - errors['L24']
            for action in ACTIONS
        })
        row['oracle_action'] = choose_oracle_action(
            errors,
            max_matched_degradation,
            losses=losses,
            max_loss_degradation=max_loss_degradation,
        )
        output.append(row)
    return output


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_yaw(rotation):
    w, x, y, z = [float(value) for value in rotation]
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def standardize_prediction(record):
    return dict(
        class_name=str(record['detection_name']),
        translation=tuple(float(value) for value in record['translation']),
        size=tuple(float(value) for value in record['size']),
        yaw=quaternion_yaw(record['rotation']),
        velocity=tuple(float(value) for value in record.get('velocity', (0.0, 0.0))[:2]),
        score=float(record['detection_score']),
    )


def load_branch_results(path, score_floor=0.1):
    payload = json.loads(Path(path).read_text())
    results = payload.get('results')
    if not isinstance(results, dict):
        raise ValueError(f'{path} is not a nuScenes result JSON')
    return {
        str(token): [
            standardize_prediction(item) for item in detections
            if float(item.get('detection_score', 0.0)) >= score_floor
        ]
        for token, detections in results.items()
    }


def load_progress(path):
    frames = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                frames.append(json.loads(line))
    if not frames:
        raise ValueError('collection progress contains no frames')
    tokens = [str(frame['token']) for frame in frames]
    if len(tokens) != len(set(tokens)):
        raise ValueError('collection progress contains duplicate tokens')
    return frames


def validate_token_sets(branch_results):
    reference = set(branch_results['L24'])
    mismatched = {
        action: len(set(rows) ^ reference)
        for action, rows in branch_results.items()
        if set(rows) != reference
    }
    if mismatched:
        raise ValueError(f'counterfactual token sets differ: {mismatched}')
    if not reference:
        raise ValueError('counterfactual predictions contain no tokens')
    return reference


def load_ground_truth(data_root, version, tokens):
    from nuscenes import NuScenes
    from nuscenes.eval.detection.utils import category_to_detection_name

    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    output = {}
    for token in tokens:
        boxes = []
        for annotation_token in nusc.get('sample', token)['anns']:
            annotation = nusc.get('sample_annotation', annotation_token)
            class_name = category_to_detection_name(annotation['category_name'])
            if class_name is None:
                continue
            boxes.append(dict(
                class_name=class_name,
                translation=tuple(float(value) for value in annotation['translation']),
                size=tuple(float(value) for value in annotation['size']),
                yaw=quaternion_yaw(annotation['rotation']),
                velocity=tuple(float(value) for value in np.nan_to_num(
                    nusc.box_velocity(annotation_token)[:2])),
                score=1.0,
            ))
        output[str(token)] = boxes
    return output


def load_metadata(path, tokens):
    frame = pd.read_csv(path)
    required = {'token', 'scene_token'} | set(ONLINE_FEATURES) | set(OFFLINE_FEATURES)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'metadata CSV missing columns: {sorted(missing)}')
    frame['token'] = frame['token'].astype(str)
    frame = frame[frame['token'].isin(tokens)].copy()
    if frame['token'].duplicated().any():
        raise ValueError('metadata CSV contains duplicate tokens')
    missing_tokens = set(tokens) - set(frame['token'])
    if missing_tokens:
        raise ValueError(f'metadata CSV missing {len(missing_tokens)} prediction tokens')
    columns = ['token', 'scene_token'] + list(ONLINE_FEATURES)
    return {
        row['token']: {name: row[name] for name in columns}
        for row in frame.to_dict('records')
    }, frame


def load_loss_rows(path):
    frame = pd.read_csv(path)
    required = {'token'} | {f'loss_{action}' for action in ACTIONS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'detection-loss CSV missing columns: {sorted(missing)}')
    return {
        str(row['token']): {
            action: float(row[f'loss_{action}']) for action in ACTIONS
        }
        for row in frame.to_dict('records')
    }


def teacher_forcing_residual_std(frames, metadata_frame, previous_rows):
    metadata = metadata_frame.set_index('token')
    previous_gt = None
    previous_scene = None
    residuals = []
    for frame, predicted in zip(frames, previous_rows):
        token = str(frame['token'])
        scene = str(frame.get('scene_token', ''))
        if scene == previous_scene and previous_gt is not None:
            residuals.append([
                float(predicted[name]) - float(previous_gt[name])
                for name in OFFLINE_FEATURES
            ])
        previous_gt = metadata.loc[token]
        previous_scene = scene
    if not residuals:
        return {name: 0.0 for name in OFFLINE_FEATURES}
    values = np.asarray(residuals, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    std = values.std(axis=0)
    return {name: float(value) for name, value in zip(OFFLINE_FEATURES, std)}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build four-depth counterfactual policy targets')
    parser.add_argument('--counterfactual-dir', required=True)
    parser.add_argument('--metadata-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--split', required=True, choices=('train', 'val'))
    parser.add_argument('--max-matched-degradation', type=float, default=0.05)
    parser.add_argument('--detection-loss-csv')
    parser.add_argument('--max-loss-degradation', type=float)
    parser.add_argument('--box-only', action='store_true')
    parser.add_argument('--score-floor', type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    counterfactual_dir = Path(args.counterfactual_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = counterfactual_dir / 'counterfactual_manifest.json'
    progress_path = counterfactual_dir / 'collection_progress.jsonl'
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    declared_split = manifest.get('data_split')
    if declared_split is not None and declared_split != args.split:
        raise ValueError(
            f'manifest split {declared_split!r} does not match {args.split!r}')
    frames = load_progress(progress_path)
    branch_paths = {
        action: Path(manifest['prediction_paths'][action])
        for action in ACTIONS
    }
    branch_results = {
        action: load_branch_results(path, args.score_floor)
        for action, path in branch_paths.items()
    }
    tokens = validate_token_sets(branch_results)
    if set(str(frame['token']) for frame in frames) != tokens:
        raise ValueError('progress tokens differ from prediction tokens')
    metadata, metadata_frame = load_metadata(args.metadata_csv, tokens)
    ground_truth = load_ground_truth(args.data_root, args.version, tokens)
    loss_rows = (
        load_loss_rows(args.detection_loss_csv)
        if args.detection_loss_csv else None
    )
    if loss_rows is not None and args.max_loss_degradation is None:
        raise ValueError(
            '--max-loss-degradation is required with --detection-loss-csv')
    metric_config = MetricConfig(box_only=args.box_only)
    rows = build_target_rows(
        frames,
        branch_results,
        ground_truth,
        metadata,
        metric_config=metric_config,
        max_matched_degradation=args.max_matched_degradation,
        loss_rows=loss_rows,
        max_loss_degradation=args.max_loss_degradation,
    )
    csv_path = output_dir / 'counterfactual_policy_targets.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    previous_rows = build_previous_prediction_rows(
        frames, branch_results, metadata=metadata)
    residual_std = teacher_forcing_residual_std(
        frames, metadata_frame, previous_rows)
    target_manifest = dict(
        split=args.split,
        source_manifest=str(manifest_path.resolve()),
        source_manifest_sha256=file_sha256(manifest_path),
        source_metadata=str(Path(args.metadata_csv).resolve()),
        source_metadata_sha256=file_sha256(args.metadata_csv),
        prediction_sha256={
            action: file_sha256(path) for action, path in branch_paths.items()
        },
        metric_config=vars(metric_config),
        max_matched_degradation=args.max_matched_degradation,
        max_loss_degradation=args.max_loss_degradation,
        detection_loss_csv=args.detection_loss_csv,
        sample_count=len(rows),
        class_counts=dict(Counter(row['oracle_action'] for row in rows)),
        teacher_forcing_residual_std=residual_std,
        output_csv=str(csv_path.resolve()),
        output_csv_sha256=file_sha256(csv_path),
    )
    target_manifest_path = output_dir / 'counterfactual_policy_targets_manifest.json'
    target_manifest_path.write_text(json.dumps(
        target_manifest, indent=2, sort_keys=True))
    print(f'wrote {len(rows)} counterfactual policy targets to {csv_path}')


if __name__ == '__main__':
    main()
