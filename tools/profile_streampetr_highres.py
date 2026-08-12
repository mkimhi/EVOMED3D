#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_summary(
    resolution,
    peak_allocated,
    peak_reserved,
    backbone_s,
    head_s,
    total_s,
):
    return {
        'resolution': [int(resolution[0]), int(resolution[1])],
        'peak_allocated_bytes': int(peak_allocated),
        'peak_reserved_bytes': int(peak_reserved),
        'backbone_ms': float(backbone_s) * 1000.0,
        'head_ms': float(head_s) * 1000.0,
        'total_ms': float(total_s) * 1000.0,
    }


def parse_args(argv=None):
    from mmcv import DictAction

    parser = argparse.ArgumentParser(
        description='Profile high-resolution StreamPETR CUDA timing and memory')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument(
        '--data-split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args(argv)


def validate_output_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'output directory is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_test_frame(model_kwargs):
    frame_data = {}
    for key, value in model_kwargs.items():
        if key in ('img', 'img_metas'):
            continue
        frame_data[key] = value[0][0].unsqueeze(0)
    return model_kwargs['img_metas'][0], frame_data


def _profile_frame(model, img, img_metas, frame_data):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    torch.cuda.synchronize()
    backbone_start = time.perf_counter()
    img_feats = model.extract_img_feat(img, len_queue=1, training_mode=False)
    torch.cuda.synchronize()
    backbone_s = time.perf_counter() - backbone_start

    head_data = dict(frame_data)
    head_data['img'] = img
    head_data['img_feats'] = img_feats
    torch.cuda.synchronize()
    head_start = time.perf_counter()
    model.simple_test_pts(img_metas, **head_data)
    torch.cuda.synchronize()
    head_s = time.perf_counter() - head_start

    total_s = time.perf_counter() - total_start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    resolution = tuple(int(v) for v in img.shape[-2:])
    return build_summary(
        resolution=resolution,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
        backbone_s=backbone_s,
        head_s=head_s,
        total_s=total_s,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.num_samples < 1:
        raise ValueError('--num-samples must be positive')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for high-resolution profiling')

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = validate_output_dir(args.output_dir)

    from mmcv import Config
    from mmcv.parallel.scatter_gather import scatter_kwargs
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from tools.collect_depth_counterfactuals import (
        _import_config_plugin,
        select_dataset_cfg,
    )
    from tools.eval_utils import limit_dataset

    cfg = Config.fromfile(str(config_path))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    _import_config_plugin(cfg, config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    dataset = build_dataset(select_dataset_cfg(cfg, args.data_split))
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
    model.to(args.device).eval()

    device_index = int(args.device.split(':', 1)[1]) if ':' in args.device else 0
    summaries = []
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            _, scattered_kwargs = scatter_kwargs((), data, [device_index], dim=0)
            model_kwargs = dict(scattered_kwargs[0])
            image_list = model_kwargs.pop('img')
            if not isinstance(image_list, list) or len(image_list) != 1:
                raise ValueError('expected one test-time image tensor')
            img = image_list[0]
            img_metas, frame_data = _prepare_test_frame(model_kwargs)
            summary = _profile_frame(model, img, img_metas, frame_data)
            summaries.append(summary)
            print(
                f'[{index + 1}/{len(dataset)}] '
                f'backbone={summary["backbone_ms"]:.2f}ms '
                f'head={summary["head_ms"]:.2f}ms '
                f'total={summary["total_ms"]:.2f}ms '
                f'peak_allocated={summary["peak_allocated_bytes"]}')

    summary = build_summary(
        resolution=summaries[0]['resolution'],
        peak_allocated=max(item['peak_allocated_bytes'] for item in summaries),
        peak_reserved=max(item['peak_reserved_bytes'] for item in summaries),
        backbone_s=sum(item['backbone_ms'] for item in summaries) / len(summaries) / 1000.0,
        head_s=sum(item['head_ms'] for item in summaries) / len(summaries) / 1000.0,
        total_s=sum(item['total_ms'] for item in summaries) / len(summaries) / 1000.0,
    )
    summary['num_samples'] = len(summaries)

    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(f'wrote {summary_path.resolve()}')


if __name__ == '__main__':
    main()
