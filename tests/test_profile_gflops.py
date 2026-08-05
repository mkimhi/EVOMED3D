import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_gflops import analyze_module, parse_args


def test_linear_trace():
    result = analyze_module(
        nn.Linear(4, 3, bias=False),
        (torch.zeros(2, 4),),
        image_shape=(1, 6, 3, 320, 800),
    )
    assert result['macs'] == 24
    assert result['gflops'] == pytest.approx(48 / 1e9)
    assert result['unsupported_ops'] == {}


def test_cli_defaults(tmp_path):
    args = parse_args([
        'config.py', 'model.pth', '--output-dir', str(tmp_path)])
    assert (args.device, args.num_samples, args.forced_action) == (
        'cuda:0', 1, None)
