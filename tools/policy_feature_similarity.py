#!/usr/bin/env python
import argparse
import csv
import os.path as osp
import sys
from collections import OrderedDict
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import DictAction
from mmdet3d.datasets import build_dataset

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

import projects.mmdet3d_plugin  # noqa: F401,E402
from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # noqa: E402
from tools.eval_utils import limit_dataset  # noqa: E402
from tools.feature_similarity import (  # noqa: E402
    DEFAULT_CAMERA_NAMES,
    linear_cka,
    RepDETR3DFeatureAnalysisAdapter,
    build_model_from_cfg,
    configure_frame_gap_dataset_cfg,
    limit_dataset_by_scenes,
    load_cfg,
    move_data_to_device,
    prepare_dataset_cfg,
)


DEFAULT_LAYERS = OrderedDict([
    ('L8', 'block8'),
    ('L12', 'block12'),
    ('L16', 'block16'),
    ('L24', 'final'),
])


def flatten_feature(features):
    if len(features) != 1:
        raise ValueError('raw layer similarity expects a single feature tensor per layer')
    feature = features[0]
    if feature.dim() != 5:
        raise ValueError(
            'feature must have shape [B, N_cam, C, H, W], '
            f'got {tuple(feature.shape)}')
    return feature


def camera_cosine_means(previous, current):
    if previous.shape != current.shape:
        raise ValueError(f'shape mismatch: {tuple(previous.shape)} vs {tuple(current.shape)}')
    cosine = F.cosine_similarity(previous.float(), current.float(), dim=2)
    return cosine.mean(dim=(0, 2, 3)).detach().cpu().numpy().tolist()


def camera_cka_values(previous, current):
    if previous.shape != current.shape:
        raise ValueError(f'shape mismatch: {tuple(previous.shape)} vs {tuple(current.shape)}')
    values = []
    for camera_index in range(previous.size(1)):
        prev_tokens = previous[:, camera_index].permute(0, 2, 3, 1).reshape(-1, previous.size(2))
        cur_tokens = current[:, camera_index].permute(0, 2, 3, 1).reshape(-1, current.size(2))
        values.append(float(linear_cka(prev_tokens.detach(), cur_tokens.detach()).detach().cpu()))
    return values


def _add_summary_stats(record, prefix, values):
    record[f'{prefix}_mean'] = float(np.mean(values)) if values else float('nan')
    record[f'{prefix}_min'] = float(np.min(values)) if values else float('nan')
    record[f'{prefix}_max'] = float(np.max(values)) if values else float('nan')
    record[f'{prefix}_spread'] = float(np.max(values) - np.min(values)) if values else float('nan')


def layer_similarity_record(previous_features, current_features, layer_name, camera_names, include_cka=False):
    previous = flatten_feature(previous_features)
    current = flatten_feature(current_features)
    values = camera_cosine_means(previous, current)
    record = {}
    for camera_name, value in zip(camera_names, values):
        record[f'similarity_{layer_name}_{camera_name}'] = float(value)
    record[f'similarity_{layer_name}_mean'] = float(np.mean(values)) if values else float('nan')
    if include_cka:
        cka_values = camera_cka_values(previous, current)
        for camera_name, value in zip(camera_names, cka_values):
            record[f'cka_{layer_name}_{camera_name}'] = float(value)
        _add_summary_stats(record, f'cka_{layer_name}', cka_values)
    return record


def _first_meta_value(value, default=''):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _first_meta_value(value[0], default=default)
    return value


def frame_metadata(adapter, data, frame_index):
    metas = adapter._img_metas(data)
    if isinstance(metas, (list, tuple)) and metas:
        if len(metas) > frame_index and isinstance(metas[frame_index], (list, tuple)):
            meta = metas[frame_index][0]
        elif isinstance(metas[0], (list, tuple)) and len(metas[0]) > frame_index:
            meta = metas[0][frame_index]
        elif isinstance(metas[0], dict):
            meta = metas[0]
        else:
            meta = {}
    elif isinstance(metas, dict):
        meta = metas
    else:
        meta = {}
    return meta if isinstance(meta, dict) else {}


def normalize_queue_image_data(adapter, data):
    img = adapter._img(data)
    if not torch.is_tensor(img):
        raise ValueError('policy similarity collection expects an image tensor')
    if img.dim() == 5:
        img = img.unsqueeze(0)
        data = dict(data)
        data['img'] = [img] if isinstance(data.get('img'), (list, tuple)) else img
    if img.dim() != 6:
        raise ValueError(
            'policy similarity collection expects queued image tensor [B, T, N_cam, C, H, W], '
            f'got {tuple(img.shape)}')
    return data, img


def _extract_layer_features(model, adapter, data, layer_specs, backbone_depth):
    return {
        layer_name: adapter.extract_backbone_layer_features(
            model, data, layer_id, depth=backbone_depth)
        for layer_name, layer_id in layer_specs.items()
    }


def _record_for_frame(meta, frame_index, has_previous):
    return {
        'token': meta.get('sample_idx', meta.get('token', '')),
        'scene_token': meta.get('scene_token', ''),
        'frame_number': meta.get('frame_idx', frame_index),
        'queue_frame_index': frame_index,
        'has_previous_frame': int(has_previous),
    }


def sample_records_from_batch(
    model,
    adapter,
    data,
    layer_specs,
    camera_names,
    backbone_depth=24,
    previous_state=None,
    include_cka=False,
):
    data, img = normalize_queue_image_data(adapter, data)
    if img.size(0) != 1:
        raise ValueError('policy similarity collection currently expects samples_per_gpu=1')
    queue_length = img.size(1)
    records = []
    prev_features_by_layer = None if previous_state is None else previous_state.get('features')
    prev_scene_token = None if previous_state is None else previous_state.get('scene_token')

    for frame_index in range(queue_length):
        frame_data = adapter.frame_data(data, frame_index)
        current_features_by_layer = _extract_layer_features(
            model, adapter, frame_data, layer_specs, backbone_depth)
        meta = frame_metadata(adapter, data, frame_index)
        scene_token = meta.get('scene_token', '')
        has_previous = prev_features_by_layer is not None and scene_token == prev_scene_token
        record = _record_for_frame(meta, frame_index, has_previous)
        if has_previous:
            for layer_name in layer_specs:
                record.update(layer_similarity_record(
                    prev_features_by_layer[layer_name],
                    current_features_by_layer[layer_name],
                    layer_name=layer_name,
                    camera_names=camera_names,
                    include_cka=include_cka,
                ))
        records.append(record)
        prev_features_by_layer = current_features_by_layer
        prev_scene_token = scene_token

    state = {'features': prev_features_by_layer, 'scene_token': prev_scene_token}
    return records, state


def apply_dataset_info_fallback(records, dataset, start_index):
    if not hasattr(dataset, 'data_infos'):
        return records
    for offset, record in enumerate(records):
        info_index = start_index + offset
        if info_index >= len(dataset.data_infos):
            break
        info = dataset.data_infos[info_index]
        if not record.get('token'):
            record['token'] = info.get('token', '')
        if not record.get('scene_token'):
            record['scene_token'] = info.get('scene_token', '')
        if record.get('frame_number', 0) == 0 and info.get('frame_idx') is not None:
            record['frame_number'] = int(info.get('frame_idx'))
    return records


def write_similarity_csv(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text('')
        return
    preferred = ['token', 'scene_token', 'frame_number', 'queue_frame_index', 'has_previous_frame']
    keys = set().union(*(record.keys() for record in records))
    fieldnames = [key for key in preferred if key in keys] + sorted(keys - set(preferred))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_layer_specs(values):
    if not values:
        return DEFAULT_LAYERS.copy()
    specs = OrderedDict()
    for value in values:
        if ':' in value:
            name, layer = value.split(':', 1)
        else:
            name = value.upper() if value.startswith('l') else value
            layer = value
        specs[name] = layer
    return specs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect adjacent-frame per-camera EVA layer similarity features for policy ML tables.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--adapter', default='repdetr3d', choices=['repdetr3d'])
    parser.add_argument('--data-split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--queue-length', type=int, default=2)
    parser.add_argument('--layers', nargs='*', default=None, help='Layer specs such as L8:block8 L12:block12 L16:block16 L24:final')
    parser.add_argument('--backbone-depth', type=int, default=24)
    parser.add_argument('--num-videos', type=int, default=None)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--out-csv', default='outputs/policy_signal_stats/val/policy_feature_similarity.csv')
    parser.add_argument('--include-cka', action='store_true', help='Also write patch-token linear CKA per layer/camera plus mean/min/max/spread summaries.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.queue_length < 2:
        raise ValueError('--queue-length must be at least 2')
    device = args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu'
    cfg = load_cfg(args.config, args.cfg_options)
    data_cfg, samples_per_gpu = prepare_dataset_cfg(cfg, args.data_split)
    configure_frame_gap_dataset_cfg(data_cfg, args.queue_length)
    dataset = build_dataset(data_cfg)
    dataset = limit_dataset_by_scenes(dataset, args.num_videos)
    if args.num_samples is not None:
        dataset = limit_dataset(dataset, args.num_samples)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    model = build_model_from_cfg(cfg, args.checkpoint, device)
    adapter = RepDETR3DFeatureAnalysisAdapter()
    layer_specs = parse_layer_specs(args.layers)
    camera_names = DEFAULT_CAMERA_NAMES

    records = []
    previous_state = None
    with torch.no_grad():
        for batch_index, data in enumerate(mmcv.track_iter_progress(data_loader)):
            data = move_data_to_device(data, device)
            batch_records, previous_state = sample_records_from_batch(
                model,
                adapter,
                data,
                layer_specs=layer_specs,
                camera_names=camera_names,
                backbone_depth=args.backbone_depth,
                previous_state=previous_state,
                include_cka=args.include_cka,
            )
            batch_records = apply_dataset_info_fallback(
                batch_records, dataset, start_index=batch_index)
            records.extend(batch_records)
    write_similarity_csv(records, args.out_csv)
    print(f'wrote {len(records)} frame similarity records to {args.out_csv}')


if __name__ == '__main__':
    main()
