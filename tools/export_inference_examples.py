#!/usr/bin/env python
"""Export short nuScenes inference examples for paper figures."""

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CAMERA_ORDER = (
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
)


def validate_export_counts(num_scenes, frames_per_scene):
    if num_scenes < 1:
        raise ValueError('--num-scenes must be at least 1')
    if frames_per_scene < 1:
        raise ValueError('--frames-per-scene must be at least 1')


def artifact_stem(scene_ordinal, frame_ordinal):
    return f'scene{scene_ordinal:02d}_frame{frame_ordinal:02d}'


def ordered_camera_paths(info, data_root):
    cameras = info.get('cams', {})
    missing = [name for name in CAMERA_ORDER if name not in cameras]
    if missing:
        raise ValueError(f'missing camera records: {", ".join(missing)}')
    paths = []
    for name in CAMERA_ORDER:
        path = Path(cameras[name]['data_path'])
        if path.is_absolute():
            pass
        elif path.exists():
            path = path.resolve()
        else:
            path = Path(data_root) / path
        paths.append((name, path))
    return paths


def select_scene_indices(data_infos, num_scenes, frames_per_scene):
    """Return the first frames from the first distinct scenes, in order."""
    selected = []
    scene_ordinals = {}
    frame_counts = {}
    for index, info in enumerate(data_infos):
        scene_token = info['scene_token']
        if scene_token not in scene_ordinals:
            if len(scene_ordinals) >= num_scenes:
                continue
            scene_ordinals[scene_token] = len(scene_ordinals)
            frame_counts[scene_token] = 0
        if frame_counts[scene_token] >= frames_per_scene:
            continue
        selected.append((
            index,
            scene_ordinals[scene_token],
            frame_counts[scene_token],
        ))
        frame_counts[scene_token] += 1
    return selected


def activation_to_uint8_maps(feature, num_cameras=6):
    """Convert EVA block output [camera, H, W, C] to display maps."""
    if not torch.is_tensor(feature) or feature.ndim != 4:
        raise ValueError('feature must be a four-dimensional torch tensor')
    if feature.shape[0] != num_cameras:
        raise ValueError(
            f'expected {num_cameras} camera features, got {feature.shape[0]}')

    energy = feature.detach().float().abs().mean(dim=-1).cpu().numpy()
    maps = []
    for camera_map in energy:
        low, high = np.percentile(camera_map, (1, 99))
        if high <= low:
            normalized = np.zeros_like(camera_map)
        else:
            normalized = np.clip((camera_map - low) / (high - low), 0, 1)
        maps.append(np.rint(normalized * 255).astype(np.uint8))
    return maps


def save_six_panel(images, labels, output_path, heatmap=False):
    """Save six images as a labeled 2-by-3 camera mosaic."""
    if len(images) != 6 or len(labels) != 6:
        raise ValueError('exactly six images and six labels are required')

    cell_size = (320, 180)
    canvas = Image.new('RGB', (cell_size[0] * 3, cell_size[1] * 2))
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        panel = image if isinstance(image, Image.Image) else Image.fromarray(image)
        if heatmap:
            panel = ImageOps.colorize(panel.convert('L'), '#081d58', '#ffffd9')
        panel = ImageOps.fit(panel.convert('RGB'), cell_size)
        x = index % 3 * cell_size[0]
        y = index // 3 * cell_size[1]
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + 116, y + 18), fill='black')
        draw.text((x + 4, y + 3), str(label), fill='white')

    canvas.save(str(output_path))


def parse_args():
    from mmcv import DictAction

    parser = argparse.ArgumentParser(
        description='Export short nuScenes input/prediction/feature examples')
    parser.add_argument('config', help='baseline or temporal model config')
    parser.add_argument('checkpoint', help='model checkpoint')
    parser.add_argument('--output-dir', default='inferece_example')
    parser.add_argument('--num-scenes', type=int, default=3)
    parser.add_argument('--frames-per-scene', type=int, default=5)
    parser.add_argument('--feature-layer', type=int, default=8)
    parser.add_argument('--score-thr', type=float, default=0.25)
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='config overrides in key=value form')
    return parser.parse_args()


def _import_config_plugin(cfg, config_path):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', str(Path(config_path).parent))
    module_path = str(plugin_dir).rstrip('/').replace('/', '.')
    importlib.import_module(module_path)


def _selected_dataset(dataset, selections):
    selected = copy.copy(dataset)
    selected.data_infos = [dataset.data_infos[index]
                           for index, _, _ in selections]
    if hasattr(selected, 'flag'):
        selected.flag = np.asarray(
            [scene_ordinal for _, scene_ordinal, _ in selections],
            dtype=np.int64)
    return selected


def _prediction_items(result):
    if isinstance(result, dict):
        if 'bbox_results' not in result:
            raise ValueError('model result dict has no bbox_results')
        result = result['bbox_results']
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise ValueError('export requires one prediction item per batch')
    return list(result)


def _result_json_path(result_files):
    if isinstance(result_files, str):
        return Path(result_files)
    if 'pts_bbox' in result_files:
        return Path(result_files['pts_bbox'])
    if len(result_files) == 1:
        return Path(next(iter(result_files.values())))
    raise ValueError(f'cannot choose prediction result from {result_files}')


def _policy_state(model):
    controller = getattr(model, 'temporal_memory_controller', None)
    if controller is None:
        return 'L24', False
    return controller.last_action or 'L24', bool(controller.last_action_forced)


def main():
    args = parse_args()
    validate_export_counts(args.num_scenes, args.frames_per_scene)
    if not 0 <= args.score_thr <= 1:
        raise ValueError('--score-thr must be between 0 and 1')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for detector inference')

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f'config not found: {config_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')

    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = Config.fromfile(str(config_path))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    _import_config_plugin(cfg, config_path)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    else:
        raise ValueError('export requires one test dataset configuration')

    dataset = build_dataset(cfg.data.test)
    selections = select_scene_indices(
        dataset.data_infos, args.num_scenes, args.frames_per_scene)
    if not selections:
        raise ValueError('the test dataset contains no selectable scenes')
    selected_scene_count = len({scene for _, scene, _ in selections})
    if selected_scene_count < args.num_scenes:
        raise ValueError(
            f'requested {args.num_scenes} scenes, found {selected_scene_count}')
    dataset = _selected_dataset(dataset, selections)
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
    blocks = getattr(model.img_backbone, 'blocks', None)
    if blocks is None:
        raise ValueError('the image backbone does not expose EVA-style blocks')
    if not 1 <= args.feature_layer <= len(blocks):
        raise ValueError(
            f'--feature-layer must be in [1, {len(blocks)}], '
            f'got {args.feature_layer}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(cfg.get('data_root', './data/nuscenes')).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f'nuScenes data root not found: {data_root}')

    capture = {'first': None, 'calls': 0}

    def capture_feature(_module, _inputs, output):
        capture['calls'] += 1
        if capture['first'] is None:
            if isinstance(output, (list, tuple)):
                output = output[0]
            capture['first'] = output.detach().cpu()

    hook = blocks[args.feature_layer - 1].register_forward_hook(capture_feature)
    parallel_model = MMDataParallel(model.cuda(), device_ids=[0])
    parallel_model.eval()
    predictions = []
    manifest_frames = []
    try:
        for item_index, data in enumerate(data_loader):
            dataset_index, scene_ordinal, frame_ordinal = selections[item_index]
            info = dataset.data_infos[item_index]
            stem = artifact_stem(scene_ordinal, frame_ordinal)
            camera_paths = ordered_camera_paths(info, data_root)
            missing = [str(path) for _, path in camera_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f'missing camera images for {info["token"]}: {missing}')

            raw_images = []
            for _, path in camera_paths:
                with Image.open(path) as image:
                    raw_images.append(image.convert('RGB').copy())
            save_six_panel(
                raw_images, CAMERA_ORDER, output_dir / f'{stem}_input.png')

            capture['first'] = None
            capture['calls'] = 0
            with torch.no_grad():
                result = parallel_model(return_loss=False, rescale=True, **data)
            predictions.extend(_prediction_items(result))
            if capture['first'] is None:
                raise RuntimeError(
                    f'feature hook did not run for sample {info["token"]}')
            feature_maps = activation_to_uint8_maps(capture['first'])
            save_six_panel(
                feature_maps,
                CAMERA_ORDER,
                output_dir / f'{stem}_feature_L{args.feature_layer}.png',
                heatmap=True,
            )
            action, forced = _policy_state(model)
            manifest_frames.append({
                'stem': stem,
                'sample_token': info['token'],
                'scene_token': info['scene_token'],
                'dataset_index': int(dataset_index),
                'camera_paths': [str(path) for _, path in camera_paths],
                'policy_action': action,
                'policy_action_forced': forced,
                'feature_hook_calls': int(capture['calls']),
            })
            print(f'[{item_index + 1}/{len(dataset)}] inferred {stem} '
                  f'action={action}')
    finally:
        hook.remove()

    result_files, _ = dataset.format_results(
        predictions,
        jsonfile_prefix=str(output_dir / '_formatted_predictions'))
    prediction_json = _result_json_path(result_files).resolve()

    from tools.visual_nuscenes import NuScenes
    import matplotlib.pyplot as plt

    gt_nusc = NuScenes(
        version='v1.0-trainval', dataroot=str(data_root),
        verbose=False, pred=False, annotations='sample_annotation')
    pred_nusc = NuScenes(
        version='v1.0-trainval', dataroot=str(data_root),
        verbose=False, pred=True,
        annotations=str(prediction_json.with_suffix('')),
        score_thr=args.score_thr)
    for frame in manifest_frames:
        token = frame['sample_token']
        stem = frame['stem']
        gt_nusc.render_sample(
            token, out_path=str(output_dir / f'{stem}_gt.png'), verbose=False)
        plt.close('all')
        pred_nusc.render_sample(
            token, out_path=str(output_dir / f'{stem}_pred.png'), verbose=False)
        plt.close('all')

    manifest = {
        'config': str(config_path.resolve()),
        'checkpoint': str(checkpoint_path.resolve()),
        'feature_layer': args.feature_layer,
        'score_threshold': args.score_thr,
        'num_scenes': args.num_scenes,
        'frames_per_scene': args.frames_per_scene,
        'prediction_json': str(prediction_json),
        'frames': manifest_frames,
    }
    with (output_dir / 'manifest.json').open('w') as stream:
        json.dump(manifest, stream, indent=2)
    print(f'Wrote {len(manifest_frames) * 4} images to {output_dir.resolve()}')


if __name__ == '__main__':
    os.environ.setdefault('LOCAL_RANK', '0')
    main()
