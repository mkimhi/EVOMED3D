#!/usr/bin/env python
import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


DEFAULT_CLASSES = (
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
)


def transform_centers_to_global(info, centers):
    if len(centers) == 0:
        return centers.reshape(0, 2)
    xyz = np.asarray(centers[:, :3], dtype=np.float32)
    lidar_rot = Quaternion(info.get('lidar2ego_rotation', [1.0, 0.0, 0.0, 0.0])).rotation_matrix
    lidar_trans = np.asarray(info.get('lidar2ego_translation', [0.0, 0.0, 0.0]), dtype=np.float32)
    ego_rot = Quaternion(info.get('ego2global_rotation', [1.0, 0.0, 0.0, 0.0])).rotation_matrix
    ego_trans = np.asarray(info.get('ego2global_translation', [0.0, 0.0, 0.0]), dtype=np.float32)
    ego = xyz @ lidar_rot.T + lidar_trans
    global_xyz = ego @ ego_rot.T + ego_trans
    return global_xyz[:, :2]


def load_infos(path):
    with open(path, 'rb') as stream:
        payload = pickle.load(stream)
    infos = payload['infos'] if isinstance(payload, dict) and 'infos' in payload else payload
    by_token = {}
    detection_classes = set(DEFAULT_CLASSES)
    for idx, info in enumerate(infos):
        token = info.get('token') or info.get('sample_token')
        if token is None:
            continue
        boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))), dtype=np.float32)
        names_all = [str(x) for x in info.get('gt_names', [])]
        keep = [i for i, name in enumerate(names_all) if name in detection_classes]
        boxes = boxes[keep] if len(boxes) else boxes
        names = [names_all[i] for i in keep]
        by_token[token] = dict(
            index=idx,
            token=token,
            scene_token=info.get('scene_token', ''),
            frame_idx=int(info.get('frame_idx', idx)),
            timestamp=int(info.get('timestamp', 0) or 0),
            gt_boxes=boxes,
            gt_global_xy=transform_centers_to_global(info, boxes[:, :3]) if len(boxes) else np.zeros((0, 2), dtype=np.float32),
            gt_names=names,
        )
    return by_token


def load_results(path, score_thr):
    with open(path) as stream:
        payload = json.load(stream)
    results = {}
    for token, preds in payload.get('results', {}).items():
        kept = []
        for pred in preds:
            score = float(pred.get('detection_score', 0.0))
            if score < score_thr:
                continue
            kept.append(dict(
                name=str(pred.get('detection_name', '')),
                score=score,
                xy=np.asarray(pred.get('translation', [0.0, 0.0])[:2], dtype=np.float32),
            ))
        results[token] = kept
    return results


def greedy_match(gt_xy, gt_names, preds, match_dist):
    gt_xy = np.asarray(gt_xy, dtype=np.float32) if len(gt_xy) else np.zeros((0, 2), dtype=np.float32)
    matched_gt = set()
    matched_pred = set()
    distances = []
    order = sorted(range(len(preds)), key=lambda i: preds[i]['score'], reverse=True)
    for pred_idx in order:
        pred = preds[pred_idx]
        best_gt = None
        best_dist = None
        for gt_idx, name in enumerate(gt_names):
            if gt_idx in matched_gt or name != pred['name']:
                continue
            dist = float(np.linalg.norm(gt_xy[gt_idx] - pred['xy']))
            if dist <= match_dist and (best_dist is None or dist < best_dist):
                best_gt = gt_idx
                best_dist = dist
        if best_gt is None:
            continue
        matched_gt.add(best_gt)
        matched_pred.add(pred_idx)
        distances.append(best_dist)
    return matched_gt, matched_pred, distances


def summarize_class(counter, prefix, names):
    for name in names:
        yield {
            'class': name,
            f'{prefix}_gt': counter[(name, 'gt')],
            f'{prefix}_tp': counter[(name, 'tp')],
            f'{prefix}_fn': counter[(name, 'fn')],
            f'{prefix}_fp': counter[(name, 'fp')],
        }


def main():
    parser = argparse.ArgumentParser(description='Rank temporal-memory failure cases against nuScenes GT and an optional baseline.')
    parser.add_argument('--ann-file', default='data/nuscenes/nuscenes2d_temporal_infos_val.pkl')
    parser.add_argument('--method-results', required=True)
    parser.add_argument('--baseline-results', default=None)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--score-thr', type=float, default=0.1)
    parser.add_argument('--match-dist', type=float, default=2.0)
    parser.add_argument('--top-k', type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    infos = load_infos(args.ann_file)
    method = load_results(args.method_results, args.score_thr)
    baseline = load_results(args.baseline_results, args.score_thr) if args.baseline_results else None

    class_names = set(DEFAULT_CLASSES)
    rows = []
    method_counts = Counter()
    baseline_counts = Counter()
    regression_counts = Counter()

    for token, info in infos.items():
        gt_boxes = info['gt_boxes']
        gt_xy = info['gt_global_xy']
        gt_names = info['gt_names']
        class_names.update(gt_names)
        method_preds = method.get(token, [])
        method_gt, method_pred, method_dists = greedy_match(gt_xy, gt_names, method_preds, args.match_dist)
        base_gt = set()
        base_pred = set()
        base_dists = []
        if baseline is not None:
            base_gt, base_pred, base_dists = greedy_match(gt_xy, gt_names, baseline.get(token, []), args.match_dist)

        gt_counter = Counter(gt_names)
        method_tp_names = Counter(gt_names[i] for i in method_gt)
        method_fn_names = Counter(name for i, name in enumerate(gt_names) if i not in method_gt)
        method_fp_names = Counter(method_preds[i]['name'] for i in range(len(method_preds)) if i not in method_pred)
        base_tp_names = Counter(gt_names[i] for i in base_gt)
        base_fn_names = Counter(name for i, name in enumerate(gt_names) if i not in base_gt)
        base_fp_names = Counter((baseline.get(token, [])[i]['name'] for i in range(len(baseline.get(token, []))) if i not in base_pred)) if baseline is not None else Counter()
        regressed_names = Counter(gt_names[i] for i in sorted(base_gt - method_gt)) if baseline is not None else Counter()
        recovered_names = Counter(gt_names[i] for i in sorted(method_gt - base_gt)) if baseline is not None else Counter()

        for name, count in gt_counter.items():
            method_counts[(name, 'gt')] += count
            baseline_counts[(name, 'gt')] += count
        for name, count in method_tp_names.items():
            method_counts[(name, 'tp')] += count
        for name, count in method_fn_names.items():
            method_counts[(name, 'fn')] += count
        for name, count in method_fp_names.items():
            method_counts[(name, 'fp')] += count
        for name, count in base_tp_names.items():
            baseline_counts[(name, 'tp')] += count
        for name, count in base_fn_names.items():
            baseline_counts[(name, 'fn')] += count
        for name, count in base_fp_names.items():
            baseline_counts[(name, 'fp')] += count
        for name, count in regressed_names.items():
            regression_counts[(name, 'regressed_fn')] += count
        for name, count in recovered_names.items():
            regression_counts[(name, 'recovered_tp')] += count

        method_fn = len(gt_boxes) - len(method_gt)
        method_fp = len(method_preds) - len(method_pred)
        base_fn = len(gt_boxes) - len(base_gt) if baseline is not None else ''
        base_fp = len(baseline.get(token, [])) - len(base_pred) if baseline is not None else ''
        regressed_fn = len(base_gt - method_gt) if baseline is not None else 0
        recovered_tp = len(method_gt - base_gt) if baseline is not None else 0
        score = 3 * regressed_fn + method_fn + 0.25 * method_fp - recovered_tp
        rows.append(dict(
            sample_token=token,
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            frame_idx_mod4=info['frame_idx'] % 4,
            timestamp=info['timestamp'],
            gt_count=len(gt_boxes),
            method_pred_count=len(method_preds),
            method_tp=len(method_gt),
            method_fn=method_fn,
            method_fp=method_fp,
            method_mean_match_dist=(sum(method_dists) / len(method_dists)) if method_dists else '',
            baseline_tp=len(base_gt) if baseline is not None else '',
            baseline_fn=base_fn,
            baseline_fp=base_fp,
            baseline_mean_match_dist=(sum(base_dists) / len(base_dists)) if base_dists else '',
            regressed_fn=regressed_fn,
            recovered_tp=recovered_tp,
            failure_score=score,
            missed_classes=';'.join(f'{k}:{v}' for k, v in sorted(method_fn_names.items())),
            fp_classes=';'.join(f'{k}:{v}' for k, v in sorted(method_fp_names.items())),
            regressed_classes=';'.join(f'{k}:{v}' for k, v in sorted(regressed_names.items())),
            recovered_classes=';'.join(f'{k}:{v}' for k, v in sorted(recovered_names.items())),
        ))

    rows.sort(key=lambda row: row['failure_score'], reverse=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with (out_dir / 'failure_cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / 'top_failure_cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[:args.top_k])

    summary_rows = []
    for name in sorted(class_names):
        row = dict(class_name=name)
        for prefix, counter in [('method', method_counts), ('baseline', baseline_counts)]:
            gt = counter[(name, 'gt')]
            tp = counter[(name, 'tp')]
            fn = counter[(name, 'fn')]
            fp = counter[(name, 'fp')]
            row.update({
                f'{prefix}_gt': gt,
                f'{prefix}_tp': tp,
                f'{prefix}_fn': fn,
                f'{prefix}_fp': fp,
                f'{prefix}_recall': (tp / gt) if gt else '',
                f'{prefix}_precision': (tp / (tp + fp)) if (tp + fp) else '',
            })
        row['regressed_fn'] = regression_counts[(name, 'regressed_fn')]
        row['recovered_tp'] = regression_counts[(name, 'recovered_tp')]
        summary_rows.append(row)
    with (out_dir / 'class_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = dict(
        ann_file=args.ann_file,
        method_results=args.method_results,
        baseline_results=args.baseline_results,
        score_thr=args.score_thr,
        match_dist=args.match_dist,
        num_samples=len(rows),
        outputs={
            'failure_cases_csv': str((out_dir / 'failure_cases.csv').resolve()),
            'top_failure_cases_csv': str((out_dir / 'top_failure_cases.csv').resolve()),
            'class_summary_csv': str((out_dir / 'class_summary.csv').resolve()),
        },
    )
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
