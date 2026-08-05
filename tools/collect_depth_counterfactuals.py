#!/usr/bin/env python
import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ACTIONS = ('L8', 'L12', 'L16', 'L24')
CONTROLLER_FIELDS = (
    'frame_index', 'frames_since_full', 'scene_token', 'memory',
    'last_action', 'last_action_forced', 'last_distillation_loss',
    '_policy_stats',
)
HEAD_FIELDS = (
    'memory_embedding', 'memory_reference_point', 'memory_timestamp',
    'memory_egopose', 'memory_velo', 'latest_routing_metadata',
)
MODEL_FIELDS = ('prev_scene_token', 'test_flag')


def clone_runtime_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_runtime_value(item) for item in value)
    return copy.deepcopy(value)


def _capture_fields(owner, fields):
    return {
        name: clone_runtime_value(getattr(owner, name))
        for name in fields if hasattr(owner, name)
    }


def capture_runtime_state(model):
    controller = model.temporal_memory_controller
    head = model.pts_bbox_head
    dynamic_names = {
        name for name in vars(model) if name.startswith('_tm_policy_prev')
    }
    model_fields = set(MODEL_FIELDS) | dynamic_names
    return {
        'controller': _capture_fields(controller, CONTROLLER_FIELDS),
        'head': _capture_fields(head, HEAD_FIELDS),
        'model': _capture_fields(model, model_fields),
        'dynamic_model_names': dynamic_names,
    }


def _restore_fields(owner, values):
    for name, value in values.items():
        setattr(owner, name, clone_runtime_value(value))


def restore_runtime_state(model, snapshot):
    _restore_fields(model.temporal_memory_controller, snapshot['controller'])
    _restore_fields(model.pts_bbox_head, snapshot['head'])
    current_dynamic = {
        name for name in vars(model) if name.startswith('_tm_policy_prev')
    }
    for name in current_dynamic - snapshot['dynamic_model_names']:
        delattr(model, name)
    _restore_fields(model, snapshot['model'])


class ForcedFramePolicy:
    def __init__(self, action):
        if action not in ACTIONS:
            raise ValueError(f'unsupported action: {action}')
        self.action = action

    def decide(self, has_memory, is_reset_frame=False, **kwargs):
        return 'L24' if is_reset_frame or not has_memory else self.action


@contextmanager
def forced_action(controller, action):
    original_policy = controller.policy
    original_age_limit = controller.max_frames_between_full
    controller.policy = ForcedFramePolicy(action)
    controller.max_frames_between_full = None
    try:
        yield
    finally:
        controller.policy = original_policy
        controller.max_frames_between_full = original_age_limit


def collect_frame_branches(model, run_inference):
    pre_state = capture_runtime_state(model)
    canonical_prediction = run_inference(None)
    canonical_action = model.temporal_memory_controller.last_action
    if canonical_action not in ACTIONS:
        raise ValueError(f'deployed policy returned unsupported action: {canonical_action}')
    canonical_post_state = capture_runtime_state(model)
    predictions = {canonical_action: canonical_prediction}
    try:
        for action in ACTIONS:
            if action == canonical_action:
                continue
            restore_runtime_state(model, pre_state)
            with forced_action(model.temporal_memory_controller, action):
                predictions[action] = run_inference(action)
    finally:
        restore_runtime_state(model, canonical_post_state)
    return predictions, canonical_action


def validate_output_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'output directory is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_test_dataset_cfg(data_cfg):
    data_cfg = copy.deepcopy(data_cfg)
    configs = data_cfg if isinstance(data_cfg, (list, tuple)) else (data_cfg,)
    for dataset_cfg in configs:
        dataset_cfg['test_mode'] = True
        dataset_cfg.pop('samples_per_gpu', None)
        if isinstance(dataset_cfg.get('dataset'), dict):
            dataset_cfg['dataset'] = prepare_test_dataset_cfg(
                dataset_cfg['dataset'])
        if isinstance(dataset_cfg.get('datasets'), (list, tuple)):
            dataset_cfg['datasets'] = [
                prepare_test_dataset_cfg(item)
                for item in dataset_cfg['datasets']
            ]
    return data_cfg


def select_dataset_cfg(cfg, split):
    data = cfg.data
    selected = data[split] if isinstance(data, dict) else getattr(data, split)
    return prepare_test_dataset_cfg(selected)


def _normalized_projector_keys(projector):
    projectors = getattr(projector, 'projectors', {})
    normalized = set()
    for key in projectors.keys():
        layer = str(key).lower()
        for prefix in ('block', 'layer', 'l'):
            if layer.startswith(prefix):
                layer = layer[len(prefix):]
                break
        normalized.add(f'block{layer}')
    return normalized


def validate_model(model):
    controller = getattr(model, 'temporal_memory_controller', None)
    head = getattr(model, 'pts_bbox_head', None)
    projector = getattr(model, 'temporal_memory_projector', None)
    if controller is None or head is None or projector is None:
        raise ValueError('collector requires an enabled temporal-memory RepDETR3D model')
    missing = {'block8', 'block12', 'block16'} - _normalized_projector_keys(projector)
    if missing:
        raise ValueError(f'temporal projector is missing layers: {sorted(missing)}')


def validate_token_sets(token_sets):
    reference = set(token_sets['L24'])
    mismatched = {
        action: len(set(tokens) ^ reference)
        for action, tokens in token_sets.items()
        if set(tokens) != reference
    }
    if mismatched:
        raise ValueError(f'prediction token sets differ from L24: {mismatched}')
    if not reference:
        raise ValueError('prediction files contain no sample tokens')
    return reference


def token_set_hash(tokens):
    payload = '\n'.join(sorted(tokens)).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    run_id,
    config_path,
    checkpoint_path,
    prediction_paths,
    tokens,
    action_counts,
    cfg_options,
    config_sha256,
    checkpoint_sha256,
    data_split='val',
):
    return {
        'shared_pre_frame_state': True,
        'run_id': run_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'config': str(Path(config_path).resolve()),
        'checkpoint': str(Path(checkpoint_path).resolve()),
        'config_sha256': config_sha256,
        'checkpoint_sha256': checkpoint_sha256,
        'canonical_state_advance_mode': 'deployed_policy',
        'canonical_state_advance_action': 'deployed_policy',
        'prediction_paths': {
            action: str(Path(path).resolve())
            for action, path in prediction_paths.items()
        },
        'sample_count': len(tokens),
        'token_set_sha256': token_set_hash(tokens),
        'canonical_action_counts': dict(action_counts),
        'snapshot_fields': {
            'controller': list(CONTROLLER_FIELDS),
            'head': list(HEAD_FIELDS),
            'model': list(MODEL_FIELDS) + ['_tm_policy_prev*'],
        },
        'cfg_options': dict(cfg_options or {}),
        'data_split': data_split,
    }


def parse_args():
    from mmcv import DictAction

    parser = argparse.ArgumentParser(
        description='Collect paired L8/L12/L16/L24 RepDETR3D predictions')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output-dir', default='outputs/drift_depth_analysis')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--data-split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _import_config_plugin(cfg, config_path):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', str(Path(config_path).parent))
    importlib.import_module(str(plugin_dir).rstrip('/').replace('/', '.'))


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
    raise ValueError(f'cannot select result JSON from {result_files}')


def _format_predictions(dataset, predictions, output_dir):
    prediction_paths = {}
    token_sets = {}
    staging = output_dir / '_formatted'
    for action in ACTIONS:
        result_files, _ = dataset.format_results(
            predictions[action],
            jsonfile_prefix=str(staging / action),
        )
        source = _result_json_path(result_files)
        destination = output_dir / f'predictions_{action}' / 'results_nusc.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        payload = json.loads(destination.read_text())
        token_sets[action] = set(payload.get('results', {}))
        prediction_paths[action] = destination
    tokens = validate_token_sets(token_sets)
    shutil.rmtree(staging, ignore_errors=True)
    return prediction_paths, tokens


def main():
    args = parse_args()
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError('--num-samples must be positive')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('counterfactual collection requires a CUDA device')

    output_dir = validate_output_dir(args.output_dir)
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from tools.eval_utils import limit_dataset

    cfg = Config.fromfile(str(config_path))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    _import_config_plugin(cfg, config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    data_cfg = select_dataset_cfg(cfg, args.data_split)
    dataset = build_dataset(data_cfg)
    if args.num_samples is not None:
        dataset = limit_dataset(dataset, args.num_samples)
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
    validate_model(model)

    device_index = int(args.device.split(':', 1)[1]) if ':' in args.device else 0
    parallel_model = MMDataParallel(model.to(args.device), device_ids=[device_index])
    predictions = {action: [] for action in ACTIONS}
    action_counts = Counter()
    progress_path = output_dir / 'collection_progress.jsonl'

    with progress_path.open('w') as progress, torch.no_grad():
        for index, data in enumerate(data_loader):
            def run_inference(_forced_action):
                result = parallel_model(return_loss=False, rescale=True, **data)
                return _prediction_items(result)

            branch_predictions, canonical_action = collect_frame_branches(
                model, run_inference)
            for action in ACTIONS:
                predictions[action].extend(branch_predictions[action])
            action_counts[canonical_action] += 1
            info = dataset.data_infos[index]
            token = info.get('token', info.get('sample_idx', str(index)))
            record = {
                'index': index,
                'token': token,
                'scene_token': info.get('scene_token', ''),
                'canonical_action': canonical_action,
            }
            progress.write(json.dumps(record) + '\n')
            progress.flush()
            print(f'[{index + 1}/{len(dataset)}] {token} canonical={canonical_action}', flush=True)

    prediction_paths, tokens = _format_predictions(
        dataset, predictions, output_dir)
    run_id = datetime.now(timezone.utc).strftime('depth-counterfactual-%Y%m%dT%H%M%SZ')
    manifest = build_manifest(
        run_id=run_id,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        prediction_paths=prediction_paths,
        tokens=tokens,
        action_counts=action_counts,
        cfg_options=args.cfg_options,
        config_sha256=file_sha256(config_path),
        checkpoint_sha256=file_sha256(checkpoint_path),
        data_split=args.data_split,
    )
    (output_dir / 'counterfactual_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    summary = {
        'run_id': run_id,
        'sample_count': len(tokens),
        'canonical_action_counts': dict(action_counts),
        'prediction_bytes': {
            action: path.stat().st_size
            for action, path in prediction_paths.items()
        },
    }
    (output_dir / 'collection_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    print(f'wrote {len(tokens)} paired prediction records to {output_dir.resolve()}')


if __name__ == '__main__':
    main()
