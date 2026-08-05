import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gflops_utils import (
    deformable_attention_macs,
    flash_attention_macs,
    macs_to_gflops,
    summarize_frames,
    validate_input_shape,
    validate_unsupported_ops,
    weighted_path_gflops,
)


def test_two_flops_per_mac():
    assert macs_to_gflops(2_300_500_000_000) == pytest.approx(4601.0)


def test_expected_image_shape():
    assert validate_input_shape((1, 6, 3, 320, 800)) == (
        1, 6, 3, 320, 800)
    with pytest.raises(ValueError, match='expected image shape'):
        validate_input_shape((1, 6, 3, 800, 1333))


def test_weighted_paths():
    assert weighted_path_gflops(
        {'L8': 100, 'L24': 400}, {'L8': 3, 'L24': 1}
    ) == pytest.approx(175)


def test_frame_summary():
    with pytest.raises(ValueError, match='no frame measurements'):
        summarize_frames([])
    value = summarize_frames([
        {'action': 'L8', 'macs': 50e9},
        {'action': 'L24', 'macs': 200e9},
    ])
    assert value['mean_gflops'] == pytest.approx(250)
    assert value['action_counts'] == {'L8': 1, 'L24': 1}


def test_unsupported_compute_is_rejected():
    with pytest.raises(ValueError, match='unsupported compute operators'):
        validate_unsupported_ops({'aten::einsum': 2})
    assert validate_unsupported_ops({'aten::reshape': 3}) == {
        'aten::reshape': 3}


def test_flash_attention_macs_counts_qk_and_attention_value_products():
    assert flash_attention_macs(
        q_shape=(400, 16, 64),
        kv_shape=(400, 2, 16, 64),
        cu_q_shape=(2,),
    ) == 2 * 16 * 400 * 400 * 64


def test_deformable_attention_macs_counts_sampling_and_weighting():
    assert deformable_attention_macs(
        value_shape=(6, 850, 8, 32),
        sampling_shape=(6, 900, 8, 4, 13, 2),
    ) == 5 * 6 * 900 * 8 * 4 * 13 * 32


def test_non_mac_tensor_ops_are_accepted():
    ops = {
        'aten::add': 1,
        'aten::gelu': 1,
        'aten::mean': 1,
        'aten::mul': 1,
        'aten::softmax': 1,
        'aten::upsample_bicubic2d': 1,
    }
    assert validate_unsupported_ops(ops) == ops
