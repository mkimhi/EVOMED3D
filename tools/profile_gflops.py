#!/usr/bin/env python
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape
import torch
from torch import nn

from tools.gflops_utils import (
    deformable_attention_macs,
    flash_attention_macs,
    macs_to_gflops,
    summarize_frames,
    validate_input_shape,
    validate_unsupported_ops,
)


ACTIONS = ('L8', 'L12', 'L16', 'L24')


class DetectorTraceWrapper(nn.Module):
    def __init__(self, model, model_kwargs):
        super().__init__()
        self.model = model
        self.model_kwargs = model_kwargs

    def forward(self, img):
        result = self.model(
            return_loss=False,
            rescale=True,
            img=[img],
            **self.model_kwargs,
        )
        prediction = result[0]['pts_bbox']
        return prediction['boxes_3d'].tensor, prediction['scores_3d']


def parse_args(argv=None):
    from mmcv import DictAction

    parser = argparse.ArgumentParser(
        description='Profile end-to-end RepDETR3D MACs/GFLOPs')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--forced-action', choices=ACTIONS)
    parser.add_argument(
        '--data-split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args(argv)


def analyze_module(module, inputs, image_shape):
    shape = validate_input_shape(image_shape)
    analysis = FlopCountAnalysis(module, inputs)
    analysis.tracer_warnings('none')
    analysis.set_op_handle(
        'prim::PythonOp.FlashAttnVarlenKVPackedFunc',
        _flash_attention_handle,
        'prim::PythonOp.MultiScaleDeformableAttnFunction',
        _deformable_attention_handle,
    )
    macs = float(analysis.total())
    unsupported = dict(analysis.unsupported_ops())
    validate_unsupported_ops(unsupported)
    return {
        'macs': macs,
        'gflops': macs_to_gflops(macs),
        'image_shape': list(shape),
        'unsupported_ops': unsupported,
        'by_operator_macs': dict(analysis.by_operator()),
    }


def _flash_attention_handle(inputs, _outputs):
    return flash_attention_macs(
        get_shape(inputs[0]),
        get_shape(inputs[1]),
        get_shape(inputs[2]),
    )


def _deformable_attention_handle(inputs, _outputs):
    return deformable_attention_macs(
        get_shape(inputs[0]),
        get_shape(inputs[3]),
    )


def frame_record(index, token, scene_token, requested_action,
                 observed_action, forced, analysis):
    row = dict(analysis)
    row.update(
        index=int(index),
        sample_token=str(token),
        scene_token=str(scene_token),
        requested_action=requested_action,
        action=str(observed_action),
        forced=bool(forced),
    )
    return row


def append_jsonl(path, record):
    with Path(path).open('a') as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write('\n')


def main(argv=None):
    args = parse_args(argv)
    if args.num_samples < 1:
        raise ValueError('--num-samples must be positive')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('GFLOPs profiling requires a CUDA device')

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    from mmcv import Config
    from mmcv.parallel.scatter_gather import scatter_kwargs
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from tools.collect_depth_counterfactuals import (
        _import_config_plugin,
        file_sha256,
        forced_action,
        select_dataset_cfg,
        validate_model,
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
    validate_model(model)

    device_index = int(args.device.split(':', 1)[1]) if ':' in args.device else 0
    frames_path = output_dir / 'frames.jsonl'
    rows = []
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            _, scattered_kwargs = scatter_kwargs(
                (), data, [device_index], dim=0)
            model_kwargs = dict(scattered_kwargs[0])
            image_list = model_kwargs.pop('img')
            if not isinstance(image_list, list) or len(image_list) != 1:
                raise ValueError('expected one test-time image tensor')
            image = image_list[0]
            shape = validate_input_shape(image.shape)
            wrapper = DetectorTraceWrapper(model, model_kwargs)
            controller = model.temporal_memory_controller
            context = (
                forced_action(controller, args.forced_action)
                if args.forced_action else nullcontext()
            )
            with context:
                analysis = analyze_module(wrapper, (image,), shape)
            observed_action = controller.last_action or 'L24'
            action_forced = bool(controller.last_action_forced)
            if (
                args.forced_action
                and observed_action != args.forced_action
                and not (observed_action == 'L24' and action_forced)
            ):
                raise RuntimeError(
                    f'requested {args.forced_action}, observed {observed_action}')
            info = dataset.data_infos[index]
            row = frame_record(
                index=index,
                token=info.get('token', info.get('sample_idx', index)),
                scene_token=info.get('scene_token', ''),
                requested_action=args.forced_action,
                observed_action=observed_action,
                forced=action_forced,
                analysis=analysis,
            )
            rows.append(row)
            append_jsonl(frames_path, row)
            print(
                f'[{index + 1}/{len(dataset)}] action={observed_action} '
                f'GFLOPs={analysis["gflops"]:.3f}')

    summary = summarize_frames(rows)
    summary.update({
        'config': str(config_path.resolve()),
        'checkpoint': str(checkpoint_path.resolve()),
        'config_sha256': file_sha256(config_path),
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'data_split': args.data_split,
        'image_shape': list(validate_input_shape(rows[0]['image_shape'])),
        'scope': 'end-to-end detector inference per six-camera timestamp',
        'counting_convention': 'one multiply-add equals two FLOPs',
        'forced_action': args.forced_action,
        'frames_jsonl': str(frames_path.resolve()),
    })
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(f'wrote {summary_path.resolve()}')


if __name__ == '__main__':
    main()
