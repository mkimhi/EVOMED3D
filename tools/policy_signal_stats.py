#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path

import mmcv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


SPEED_THRESHOLD_SETS = {
    'conservative': (1.0, 5.0, 10.0),
    'default': (2.0, 8.0, 12.0),
    'aggressive': (3.0, 10.0, 15.0),
}

YAW_THRESHOLD_SETS = {
    'conservative': (2.0, 5.0, 10.0),
    'default': (3.0, 7.0, 12.0),
    'aggressive': (5.0, 10.0, 15.0),
}

ACTIONS = ('L8', 'L12', 'L16', 'L24')
NEAR_DISTANCE_THRESHOLDS = (10, 20, 30, 50)


def yaw_from_quaternion(rotation):
    w, x, y, z = [float(v) for v in rotation]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_delta_degrees(prev_yaw, cur_yaw):
    delta = (cur_yaw - prev_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(delta)


def quaternion_rotation_matrix(rotation):
    w, x, y, z = [float(v) for v in rotation]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_lidar_centers_to_global(info, centers):
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if centers.size == 0:
        return centers
    lidar_rot = quaternion_rotation_matrix(info.get('lidar2ego_rotation', [1, 0, 0, 0]))
    lidar_trans = np.asarray(info.get('lidar2ego_translation', [0, 0, 0]), dtype=np.float64)
    ego_rot = quaternion_rotation_matrix(info['ego2global_rotation'])
    ego_trans = np.asarray(info['ego2global_translation'], dtype=np.float64)
    ego_centers = centers @ lidar_rot.T + lidar_trans
    return ego_centers @ ego_rot.T + ego_trans


def _valid_mask(info, include_invalid=False):
    boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))))
    if boxes.size == 0:
        return boxes.reshape(0, 7), np.zeros((0,), dtype=bool)
    if include_invalid or 'valid_flag' not in info:
        return boxes, np.ones((len(boxes),), dtype=bool)
    return boxes, np.asarray(info['valid_flag'], dtype=bool)


def valid_boxes_and_names(info, include_invalid=False):
    boxes, mask = _valid_mask(info, include_invalid=include_invalid)
    names = np.asarray(info.get('gt_names', np.array([''] * len(boxes))))
    return boxes[mask], names[mask]


def annotation_stats(info, include_invalid=False):
    stats = {}
    boxes, mask = _valid_mask(info, include_invalid=include_invalid)
    boxes = boxes[mask]
    stats['num_annotations'] = int(len(boxes))
    if len(boxes) == 0:
        for key in [
            'avg_object_distance_m', 'median_object_distance_m', 'min_object_distance_m',
            'p10_object_distance_m', 'p90_object_distance_m', 'avg_box_length_m',
            'avg_box_width_m', 'avg_box_height_m', 'avg_box_volume_m3', 'avg_box_area_m2',
            'avg_object_speed_mps', 'max_object_speed_mps']:
            stats[key] = float('nan')
        for threshold in NEAR_DISTANCE_THRESHOLDS:
            stats[f'near_object_count_{threshold}m'] = 0
        stats.update(camera_annotation_stats(info))
        return stats

    distances = np.linalg.norm(boxes[:, :2].astype(np.float64), axis=1)
    dims = boxes[:, 3:6].astype(np.float64)
    volumes = np.prod(dims, axis=1)
    areas = dims[:, 0] * dims[:, 1]
    velocities = np.asarray(info.get('gt_velocity', np.zeros((len(mask), 2))), dtype=np.float64)
    if len(velocities) == len(mask):
        velocities = velocities[mask]
    object_speeds = np.linalg.norm(velocities.reshape(-1, 2), axis=1) if len(velocities) else np.array([])

    stats.update(dict(
        avg_object_distance_m=float(distances.mean()),
        median_object_distance_m=float(np.median(distances)),
        min_object_distance_m=float(distances.min()),
        p10_object_distance_m=float(np.percentile(distances, 10)),
        p90_object_distance_m=float(np.percentile(distances, 90)),
        avg_box_length_m=float(dims[:, 0].mean()),
        avg_box_width_m=float(dims[:, 1].mean()),
        avg_box_height_m=float(dims[:, 2].mean()),
        avg_box_volume_m3=float(volumes.mean()),
        avg_box_area_m2=float(areas.mean()),
        avg_object_speed_mps=float(object_speeds.mean()) if object_speeds.size else float('nan'),
        max_object_speed_mps=float(object_speeds.max()) if object_speeds.size else float('nan'),
    ))
    for threshold in NEAR_DISTANCE_THRESHOLDS:
        stats[f'near_object_count_{threshold}m'] = int((distances < threshold).sum())
    stats.update(camera_annotation_stats(info))
    return stats


def camera_annotation_stats(info):
    bboxes2d = info.get('bboxes2d', []) or []
    counts = [len(np.asarray(items).reshape(-1, 4)) if np.asarray(items).size else 0 for items in bboxes2d]
    areas = []
    for items in bboxes2d:
        boxes = np.asarray(items, dtype=np.float64).reshape(-1, 4)
        if len(boxes):
            wh = np.maximum(boxes[:, 2:4] - boxes[:, 0:2], 0.0)
            areas.extend((wh[:, 0] * wh[:, 1]).tolist())
    return dict(
        camera_annotation_count=int(sum(counts)),
        max_camera_annotation_count=int(max(counts)) if counts else 0,
        min_camera_annotation_count=int(min(counts)) if counts else 0,
        avg_camera_annotation_count=float(np.mean(counts)) if counts else 0.0,
        avg_2d_box_area_px=float(np.mean(areas)) if areas else float('nan'),
        max_2d_box_area_px=float(np.max(areas)) if areas else float('nan'),
    )


def new_object_proxy_stats(prev_info, info, include_invalid=False, match_distance_m=2.0):
    cur_boxes, cur_names = valid_boxes_and_names(info, include_invalid=include_invalid)
    if prev_info is None or prev_info.get('scene_token') != info.get('scene_token'):
        return dict(
            new_object_proxy_count=int(len(cur_boxes)),
            matched_object_proxy_count=0,
            disappeared_object_proxy_count=0,
            new_object_proxy_fraction=1.0 if len(cur_boxes) else 0.0,
        )

    prev_boxes, prev_names = valid_boxes_and_names(prev_info, include_invalid=include_invalid)
    cur_centers = transform_lidar_centers_to_global(info, cur_boxes[:, :3]) if len(cur_boxes) else np.zeros((0, 3))
    prev_centers = transform_lidar_centers_to_global(prev_info, prev_boxes[:, :3]) if len(prev_boxes) else np.zeros((0, 3))
    used_prev = set()
    matched = 0
    for cur_center, cur_name in zip(cur_centers, cur_names):
        best_index = None
        best_distance = float('inf')
        for prev_index, (prev_center, prev_name) in enumerate(zip(prev_centers, prev_names)):
            if prev_index in used_prev or cur_name != prev_name:
                continue
            distance = float(np.linalg.norm(cur_center[:2] - prev_center[:2]))
            if distance < best_distance:
                best_distance = distance
                best_index = prev_index
        if best_index is not None and best_distance <= match_distance_m:
            used_prev.add(best_index)
            matched += 1
    new_count = int(len(cur_boxes) - matched)
    disappeared = int(len(prev_boxes) - len(used_prev))
    return dict(
        new_object_proxy_count=new_count,
        matched_object_proxy_count=int(matched),
        disappeared_object_proxy_count=disappeared,
        new_object_proxy_fraction=float(new_count / len(cur_boxes)) if len(cur_boxes) else 0.0,
    )


def collect_policy_signal_records(
    infos,
    include_invalid=False,
    compute_new_object_proxy=True,
    new_object_match_distance_m=2.0,
):
    records = []
    prev_info = None
    prev_speed = 0.0
    prev_yaw_rate = 0.0
    frame_number = 0
    for index, info in enumerate(infos):
        scene_token = info.get('scene_token', '')
        is_scene_start = prev_info is None or scene_token != prev_info.get('scene_token')
        frame_number = 0 if is_scene_start else frame_number + 1
        timestamp = int(info.get('timestamp', 0))
        yaw = yaw_from_quaternion(info['ego2global_rotation'])
        ego_translation = np.asarray(info['ego2global_translation'], dtype=np.float64)
        speed = 0.0
        yaw_rate = 0.0
        ego_accel = 0.0
        yaw_accel = 0.0
        delta_translation = 0.0
        dt = 0.0
        if not is_scene_start:
            dt = (timestamp - int(prev_info.get('timestamp', 0))) / 1e6
            if dt > 0.0:
                prev_translation = np.asarray(prev_info['ego2global_translation'], dtype=np.float64)
                delta_translation = float(np.linalg.norm(ego_translation[:2] - prev_translation[:2]))
                speed = float(delta_translation / dt)
                prev_yaw = yaw_from_quaternion(prev_info['ego2global_rotation'])
                yaw_rate = yaw_delta_degrees(prev_yaw, yaw) / dt
                ego_accel = float((speed - prev_speed) / dt)
                yaw_accel = float((yaw_rate - prev_yaw_rate) / dt)

        record = dict(
            index=index,
            token=info.get('token', ''),
            scene_token=scene_token,
            frame_number=int(frame_number),
            info_frame_idx=int(info.get('frame_idx', frame_number)),
            is_scene_start=int(is_scene_start),
            timestamp=timestamp,
            dt_s=float(dt),
            ego_x_m=float(ego_translation[0]),
            ego_y_m=float(ego_translation[1]),
            ego_z_m=float(ego_translation[2]) if ego_translation.size > 2 else 0.0,
            ego_yaw_deg=float(math.degrees(yaw)),
            delta_ego_translation_m=float(delta_translation),
            ego_speed_mps=float(speed),
            ego_accel_mps2=float(ego_accel),
            yaw_rate_deg_s=float(yaw_rate),
            abs_yaw_rate_deg_s=float(abs(yaw_rate)),
            yaw_accel_deg_s2=float(yaw_accel),
            abs_yaw_accel_deg_s2=float(abs(yaw_accel)),
        )
        record.update(annotation_stats(info, include_invalid=include_invalid))
        if compute_new_object_proxy:
            record.update(new_object_proxy_stats(
                None if is_scene_start else prev_info,
                info,
                include_invalid=include_invalid,
                match_distance_m=new_object_match_distance_m,
            ))
        records.append(record)
        prev_info = info
        prev_speed = speed
        prev_yaw_rate = yaw_rate
    return records


def _finite(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def summarize_values(values):
    values = _finite(values)
    if values.size == 0:
        return dict(count=0)
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    summary = {
        'count': int(values.size),
        'mean': float(values.mean()),
        'std': float(values.std()),
    }
    for pct, value in zip(percentiles, np.percentile(values, percentiles)):
        summary[f'p{pct}'] = float(value)
    return summary


def action_histogram(values, thresholds):
    values = _finite(values)
    counts = dict.fromkeys(ACTIONS, 0)
    t0, t1, t2 = thresholds
    for value in values:
        if value < t0:
            counts['L8'] += 1
        elif value < t1:
            counts['L12'] += 1
        elif value < t2:
            counts['L16'] += 1
        else:
            counts['L24'] += 1
    total = max(1, int(values.size))
    return {
        action: {'count': int(count), 'fraction': float(count / total)}
        for action, count in counts.items()
    }


def make_summary(records):
    summary = {'num_frames': len(records)}
    for key in [
        'ego_speed_mps', 'abs_yaw_rate_deg_s', 'ego_accel_mps2',
        'abs_yaw_accel_deg_s2', 'avg_object_distance_m', 'num_annotations',
        'avg_box_volume_m3', 'avg_object_speed_mps', 'new_object_proxy_count',
        'camera_annotation_count', 'avg_2d_box_area_px',
    ]:
        if records and key in records[0]:
            summary[key] = summarize_values([row[key] for row in records])
    speed = [row['ego_speed_mps'] for row in records]
    yaw = [row['abs_yaw_rate_deg_s'] for row in records]
    summary['speed_policy_thresholds'] = {
        name: action_histogram(speed, thresholds)
        for name, thresholds in SPEED_THRESHOLD_SETS.items()
    }
    summary['yaw_policy_thresholds'] = {
        name: action_histogram(yaw, thresholds)
        for name, thresholds in YAW_THRESHOLD_SETS.items()
    }
    return summary


def write_records_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text('')
        return
    preferred = [
        'index', 'token', 'scene_token', 'frame_number', 'info_frame_idx', 'is_scene_start',
        'timestamp', 'dt_s', 'ego_x_m', 'ego_y_m', 'ego_z_m', 'ego_yaw_deg',
        'delta_ego_translation_m', 'ego_speed_mps', 'ego_accel_mps2',
        'yaw_rate_deg_s', 'abs_yaw_rate_deg_s', 'yaw_accel_deg_s2', 'abs_yaw_accel_deg_s2',
        'num_annotations', 'avg_object_distance_m', 'median_object_distance_m',
        'min_object_distance_m', 'p10_object_distance_m', 'p90_object_distance_m',
        'near_object_count_10m', 'near_object_count_20m', 'near_object_count_30m',
        'near_object_count_50m', 'avg_box_length_m', 'avg_box_width_m', 'avg_box_height_m',
        'avg_box_volume_m3', 'avg_box_area_m2', 'avg_object_speed_mps', 'max_object_speed_mps',
        'camera_annotation_count', 'max_camera_annotation_count', 'min_camera_annotation_count',
        'avg_camera_annotation_count', 'avg_2d_box_area_px', 'max_2d_box_area_px',
        'new_object_proxy_count', 'matched_object_proxy_count', 'disappeared_object_proxy_count',
        'new_object_proxy_fraction',
    ]
    extra = sorted(set().union(*(row.keys() for row in records)) - set(preferred))
    fieldnames = [key for key in preferred if key in records[0]] + extra
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _save_figure(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_1d_kde(values, title, xlabel, path):
    values = _finite(values)
    plt.figure(figsize=(7, 4))
    if values.size >= 2 and np.unique(values).size >= 2:
        xs = np.linspace(values.min(), values.max(), 512)
        ys = gaussian_kde(values)(xs)
        plt.plot(xs, ys, linewidth=2)
        plt.fill_between(xs, ys, alpha=0.25)
    elif values.size == 1:
        plt.axvline(values[0], linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('density')
    plt.grid(alpha=0.25)
    _save_figure(path)


def plot_2d_kde(x, y, title, xlabel, ylabel, path):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    plt.figure(figsize=(6, 5))
    if x.size >= 3 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
        order = density.argsort()
        plt.scatter(x[order], y[order], c=density[order], s=8, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('KDE density')
    else:
        plt.scatter(x, y, s=8, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    _save_figure(path)


def write_plots(records, out_dir):
    out_dir = Path(out_dir)
    speed = np.asarray([row['ego_speed_mps'] for row in records], dtype=np.float64)
    yaw = np.asarray([row['abs_yaw_rate_deg_s'] for row in records], dtype=np.float64)
    distance = np.asarray([row['avg_object_distance_m'] for row in records], dtype=np.float64)
    annotations = np.asarray([row['num_annotations'] for row in records], dtype=np.float64)

    plot_1d_kde(speed, 'Ego Speed KDE', 'ego speed (m/s)', out_dir / 'ego_speed_kde.png')
    plot_1d_kde(yaw, 'Absolute Ego Yaw Rate KDE', 'absolute yaw rate (deg/s)', out_dir / 'abs_yaw_rate_kde.png')
    plot_2d_kde(speed, distance, 'Ego Speed vs Average Object Distance', 'ego speed (m/s)', 'average object distance (m)', out_dir / 'speed_vs_avg_object_distance_kde.png')
    plot_2d_kde(speed, annotations, 'Ego Speed vs Number of Annotations', 'ego speed (m/s)', 'number of annotations', out_dir / 'speed_vs_num_annotations_kde.png')
    plot_2d_kde(yaw, distance, 'Absolute Yaw Rate vs Average Object Distance', 'absolute yaw rate (deg/s)', 'average object distance (m)', out_dir / 'yaw_vs_avg_object_distance_kde.png')
    plot_2d_kde(yaw, annotations, 'Absolute Yaw Rate vs Number of Annotations', 'absolute yaw rate (deg/s)', 'number of annotations', out_dir / 'yaw_vs_num_annotations_kde.png')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect frame-level ego, object, and camera metadata features for dynamic policy analysis.')
    parser.add_argument('--ann-file', default='data/nuscenes/nuscenes2d_temporal_infos_val.pkl', help='nuScenes temporal info pickle containing an infos list.')
    parser.add_argument('--out-dir', default='outputs/policy_signal_stats/val', help='Directory for CSV, JSON summary, and plots.')
    parser.add_argument('--include-invalid', action='store_true', help='Include boxes whose valid_flag is false in object statistics.')
    parser.add_argument('--disable-new-object-proxy', action='store_true', help='Skip same-class global-center matching against the previous frame.')
    parser.add_argument('--new-object-match-distance', type=float, default=2.0, help='Global XY distance threshold in meters for the new-object proxy matcher.')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap for quick smoke runs.')
    return parser.parse_args()


def main():
    args = parse_args()
    data = mmcv.load(args.ann_file)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    if args.max_samples is not None:
        infos = infos[:args.max_samples]

    records = collect_policy_signal_records(
        infos,
        include_invalid=args.include_invalid,
        compute_new_object_proxy=not args.disable_new_object_proxy,
        new_object_match_distance_m=args.new_object_match_distance,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(records, out_dir / 'policy_signal_stats.csv')
    summary = make_summary(records)
    summary['ann_file'] = args.ann_file
    summary['include_invalid'] = bool(args.include_invalid)
    summary['new_object_proxy_enabled'] = not args.disable_new_object_proxy
    summary['new_object_match_distance_m'] = float(args.new_object_match_distance)
    with (out_dir / 'policy_signal_summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    write_plots(records, out_dir)

    print(f'wrote {len(records)} frame records to {out_dir / "policy_signal_stats.csv"}')
    print(f'wrote summary to {out_dir / "policy_signal_summary.json"}')
    print(f'wrote plots to {out_dir}')


if __name__ == '__main__':
    main()
