#!/usr/bin/env python
import argparse

from fvcore.nn import FlopCountAnalysis

from tools.gflops_utils import (
    macs_to_gflops,
    validate_input_shape,
    validate_unsupported_ops,
)


ACTIONS = ('L8', 'L12', 'L16', 'L24')


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
