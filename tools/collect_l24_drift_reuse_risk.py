#!/usr/bin/env python
"""Collect base-L24 drift and sequential naive-reuse K=2 predictions."""

import argparse
import copy
import csv
import hashlib
import importlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.l24_drift_reuse_risk import full_feature_cosine, select_reuse_targets


EXPECTED_SCENE_COUNT = 100
EXPECTED_TARGETS_PER_SCENE = 20
EXPECTED_TARGET_COUNT = EXPECTED_SCENE_COUNT * EXPECTED_TARGETS_PER_SCENE


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('output directory is not empty: {}'.format(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_test_dataset_cfg(data_cfg):
    data_cfg = copy.deepcopy(data_cfg)
    configs = data_cfg if isinstance(data_cfg, (list, tuple)) else (data_cfg,)
    for dataset_cfg in configs:
        dataset_cfg['test_mode'] = True
        dataset_cfg.pop('samples_per_gpu', None)
    return data_cfg


def detach_cpu_features(features):
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError('base full L24 extraction must return a non-empty feature list')
    detached = []
    for feature in features:
        if not torch.is_tensor(feature):
            raise ValueError('base full L24 extraction returned a non-tensor feature')
        detached.append(feature.detach().cpu())
    return detached


def record_target_drift(token, scene_token, history_by_scene, current_features=None):
    """Build target-only dt=1/dt=2 drift, rejecting scene-crossing history."""
    history = history_by_scene.get(scene_token, [])
    if len(history) < 2:
        raise ValueError('target requires two full-L24 history frames in the same scene')
    if current_features is None:
        raise ValueError('target drift requires current full-L24 features')
    return {
        'token': token,
        'scene_token': scene_token,
        'drift_dt1': 1.0 - full_feature_cosine(history[-1], current_features),
        'drift_dt2': 1.0 - full_feature_cosine(history[-2], current_features),
    }


def build_collection_manifest(
    run_id,
    config_path,
    checkpoint_path,
    selection_path,
    drift_path,
    prediction_path,
    config_sha256,
    checkpoint_sha256,
    seed,
    scene_count,
    target_count,
    cfg_options=None,
    resolved_temporal_memory=None,
):
    """Return provenance for the fixed base-L24 / naive-reuse K=2 study."""
    return {
        'run_id': run_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'config': str(Path(config_path).resolve()),
        'checkpoint': str(Path(checkpoint_path).resolve()),
        'config_sha256': config_sha256,
        'checkpoint_sha256': checkpoint_sha256,
        'representation_source': 'base_full_l24_backbone_neck',
        'prediction_mode': 'naive_reuse_k2',
        'keyframe_interval': 2,
        'seed': int(seed),
        'selection_path': str(Path(selection_path).resolve()),
        'drift_path': str(Path(drift_path).resolve()),
        'prediction_path': str(Path(prediction_path).resolve()),
        'qualified_scene_count': int(scene_count),
        'selected_target_count': int(target_count),
        'cfg_options': dict(cfg_options or {}),
        'resolved_temporal_memory': dict(resolved_temporal_memory or {}),
    }


def _validate_fixed_target_counts(num_scenes, targets_per_scene):
    if num_scenes != EXPECTED_SCENE_COUNT or targets_per_scene != EXPECTED_TARGETS_PER_SCENE:
        raise ValueError(
            'collector requires exactly {} scenes and {} targets per scene'.format(
                EXPECTED_SCENE_COUNT, EXPECTED_TARGETS_PER_SCENE))


def _validate_temporal_memory_config(cfg):
    temporal_memory = cfg.model.get('temporal_memory', None)
    if temporal_memory is None or not temporal_memory.get('enabled', False):
        raise ValueError('collector requires an enabled temporal-memory model')
    if temporal_memory.get('updater', 'reuse') != 'reuse':
        raise ValueError('collector requires temporal_memory.updater=reuse')
    if int(temporal_memory.get('keyframe_interval', 1)) != 2:
        raise ValueError('collector requires temporal_memory.keyframe_interval=2')
    return {
        'enabled': True,
        'updater': 'reuse',
        'keyframe_interval': 2,
        'partial_layer': str(temporal_memory.get('partial_layer', 'block8')),
    }


def _import_config_plugin(cfg, config_path):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', str(Path(config_path).parent))
    importlib.import_module(str(plugin_dir).rstrip('/').replace('/', '.'))


def _restrict_dataset_to_selected_scenes(dataset, scene_tokens):
    if not hasattr(dataset, 'data_infos'):
        raise ValueError('collector requires dataset.data_infos for sequential scene selection')
    selected_infos = [
        info for info in dataset.data_infos
        if isinstance(info, dict) and info.get('scene_token') in scene_tokens
    ]
    if not selected_infos:
        raise ValueError('no frames remain after selecting qualified scenes')
    dataset.data_infos = selected_infos
    if hasattr(dataset, 'flag'):
        dataset.flag = np.asarray(dataset.flag)[:len(selected_infos)]
    dataset._eval_sample_tokens = {info['token'] for info in selected_infos}
    return selected_infos


def _write_csv(rows, path, fieldnames):
    path = Path(path)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_items(result):
    if isinstance(result, dict):
        result = result.get('bbox_results', result)
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise ValueError('collector requires exactly one prediction item per batch')
    return list(result)


def _result_json_path(result_files):
    if isinstance(result_files, str):
        return Path(result_files)
    if 'pts_bbox' in result_files:
        return Path(result_files['pts_bbox'])
    if len(result_files) == 1:
        return Path(next(iter(result_files.values())))
    raise ValueError('cannot select result JSON from {}'.format(result_files))


def _format_target_predictions(dataset, target_infos, predictions, output_dir):
    if len(target_infos) != len(predictions):
        raise ValueError('target prediction count does not match selected targets')
    original_infos = dataset.data_infos
    original_flag = getattr(dataset, 'flag', None)
    try:
        dataset.data_infos = list(target_infos)
        if original_flag is not None:
            dataset.flag = np.asarray(original_flag)[:len(target_infos)]
        staging = output_dir / '_formatted_targets'
        result_files, _ = dataset.format_results(
            predictions, jsonfile_prefix=str(staging))
        source = _result_json_path(result_files)
        destination = output_dir / 'predictions_naive_reuse_k2' / 'results_nusc.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
    finally:
        dataset.data_infos = original_infos
        if original_flag is not None:
            dataset.flag = original_flag


def _validate_prediction_coverage(prediction_path, selected_tokens):
    payload = json.loads(Path(prediction_path).read_text())
    result_tokens = set(payload.get('results', {}))
    if result_tokens != set(selected_tokens):
        raise ValueError('result JSON token coverage does not exactly match selection.csv')


def _image_tensor(data):
    image = data.get('img')
    if isinstance(image, (list, tuple)):
        if len(image) != 1:
            raise ValueError('collector requires exactly one test-time image augmentation')
        image = image[0]
    if not torch.is_tensor(image):
        raise ValueError('collector requires image tensors')
    return image


def _assert_reuse_action(model):
    controller = getattr(model, 'temporal_memory_controller', None)
    if controller is None or getattr(controller, 'updater_name', None) != 'reuse':
        raise ValueError('collector requires the temporal-memory reuse updater')
    if getattr(controller, 'last_action', None) == 'L24':
        raise ValueError('selected target did not take the naive-reuse action')


def parse_args():
    from mmcv import DictAction

    parser = argparse.ArgumentParser(
        description='Collect base-L24 drift and sequential naive-reuse K=2 predictions')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-scenes', type=int, default=EXPECTED_SCENE_COUNT)
    parser.add_argument('--targets-per-scene', type=int, default=EXPECTED_TARGETS_PER_SCENE)
    parser.add_argument('--seed', type=int, default=20260805)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    _validate_fixed_target_counts(args.num_scenes, args.targets_per_scene)
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('L24 drift collection requires a CUDA device')

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = validate_output_dir(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from mmcv import Config
    from mmcv.parallel import MMDataParallel, scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = Config.fromfile(str(config_path))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    _import_config_plugin(cfg, config_path)
    resolved_temporal_memory = _validate_temporal_memory_config(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    dataset = build_dataset(prepare_test_dataset_cfg(cfg.data.test))
    targets = select_reuse_targets(
        dataset.data_infos,
        num_scenes=args.num_scenes,
        targets_per_scene=args.targets_per_scene,
        keyframe_interval=2)
    if len(targets) != EXPECTED_TARGET_COUNT:
        raise ValueError('selection did not produce exactly {} target frames'.format(EXPECTED_TARGET_COUNT))
    selected_tokens = [row['token'] for row in targets]
    if len(selected_tokens) != len(set(selected_tokens)):
        raise ValueError('selection contains duplicate target tokens')
    selected_scene_tokens = {row['scene_token'] for row in targets}
    if len(selected_scene_tokens) != EXPECTED_SCENE_COUNT:
        raise ValueError('selection did not produce exactly {} scenes'.format(EXPECTED_SCENE_COUNT))
    selection_path = output_dir / 'selection.csv'
    _write_csv(targets, selection_path, ('token', 'scene_token', 'frame_index'))

    collection_infos = _restrict_dataset_to_selected_scenes(dataset, selected_scene_tokens)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(checkpoint_path), map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.eval()
    device_index = int(args.device.split(':', 1)[1]) if ':' in args.device else 0
    model = model.to(args.device)
    parallel_model = MMDataParallel(model, device_ids=[device_index])

    target_tokens = set(selected_tokens)
    target_predictions = []
    drift_rows = []
    history_by_scene = {}
    active_scene = None
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            info = collection_infos[index]
            token = info.get('token')
            scene_token = info.get('scene_token')
            if not token or not scene_token:
                raise ValueError('every collected frame must contain token and scene_token')
            if scene_token != active_scene:
                active_scene = scene_token
                history_by_scene = {scene_token: []}
            feature_data = scatter({'img': data['img']}, [device_index])[0]
            full_features = detach_cpu_features(
                model.extract_feat(_image_tensor(feature_data), T=1))
            result = parallel_model(return_loss=False, rescale=True, **data)
            if token in target_tokens:
                _assert_reuse_action(model)
                drift_rows.append(record_target_drift(
                    token, scene_token, history_by_scene, full_features))
                target_predictions.extend(_prediction_items(result))
            history = history_by_scene[scene_token]
            history.append(full_features)
            del history[:-2]

    if len(drift_rows) != EXPECTED_TARGET_COUNT or len(target_predictions) != EXPECTED_TARGET_COUNT:
        raise ValueError('final target collection count is not {}'.format(EXPECTED_TARGET_COUNT))
    if {row['token'] for row in drift_rows} != target_tokens:
        raise ValueError('drift rows do not exactly match selected target tokens')
    drift_path = output_dir / 'full_l24_drift.csv'
    _write_csv(drift_rows, drift_path, ('token', 'scene_token', 'drift_dt1', 'drift_dt2'))
    target_info_by_token = {info['token']: info for info in collection_infos}
    target_infos = [target_info_by_token[token] for token in selected_tokens]
    prediction_path = _format_target_predictions(
        dataset, target_infos, target_predictions, output_dir)
    _validate_prediction_coverage(prediction_path, selected_tokens)

    run_id = datetime.now(timezone.utc).strftime('l24-drift-reuse-k2-%Y%m%dT%H%M%SZ')
    manifest = build_collection_manifest(
        run_id, config_path, checkpoint_path, selection_path, drift_path,
        prediction_path, file_sha256(config_path), file_sha256(checkpoint_path),
        args.seed, len(selected_scene_tokens), len(selected_tokens),
        cfg_options=args.cfg_options,
        resolved_temporal_memory=resolved_temporal_memory)
    (output_dir / 'collection_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    summary = {
        'run_id': run_id,
        'qualified_scene_count': len(selected_scene_tokens),
        'selected_target_count': len(selected_tokens),
        'drift_record_count': len(drift_rows),
        'prediction_path': str(prediction_path.resolve()),
    }
    (output_dir / 'collection_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    print('wrote {} base-L24 drift and K=2 reuse target records to {}'.format(
        len(selected_tokens), output_dir.resolve()))


if __name__ == '__main__':
    main()
