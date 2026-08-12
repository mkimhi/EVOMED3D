import sys
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_gflops import (
    analyze_module,
    append_jsonl,
    DetectorTraceWrapper,
    frame_record,
    parse_args,
    reported_action,
    validate_profile_model,
    validate_observed_action,
)


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


def test_script_help_runs_from_repo_root():
    script = Path(__file__).resolve().parents[1] / 'tools/profile_gflops.py'
    result = subprocess.run(
        [sys.executable, str(script), '--help'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_frame_record_preserves_provenance():
    analysis = {
        'macs': 10,
        'gflops': 2e-8,
        'image_shape': [1, 6, 3, 320, 800],
        'unsupported_ops': {},
        'by_operator_macs': {'linear': 10},
    }
    row = frame_record(
        index=2,
        token='sample',
        scene_token='scene',
        requested_action='L12',
        observed_action='L12',
        forced=True,
        analysis=analysis,
    )
    assert row == {
        **analysis,
        'index': 2,
        'sample_token': 'sample',
        'scene_token': 'scene',
        'requested_action': 'L12',
        'action': 'L12',
        'forced': True,
    }


def test_append_jsonl_writes_one_record(tmp_path):
    import json

    path = tmp_path / 'frames.jsonl'
    append_jsonl(path, {'action': 'L24', 'macs': 5})
    assert json.loads(path.read_text()) == {'action': 'L24', 'macs': 5}


def test_detector_trace_wrapper_reconstructs_image_list():
    class Boxes:
        tensor = torch.ones(1, 9)

    class Detector(nn.Module):
        def forward(self, return_loss, rescale, img, marker):
            assert return_loss is False
            assert rescale is True
            assert isinstance(img, list) and len(img) == 1
            assert marker == 'kept'
            return [{
                'pts_bbox': {
                    'boxes_3d': Boxes(),
                    'scores_3d': img[0].mean().reshape(1),
                },
            }]

    image = torch.ones(1, 6, 3, 320, 800)
    boxes, scores = DetectorTraceWrapper(
        Detector(), {'marker': 'kept'})(image)
    assert boxes.shape == (1, 9)
    assert scores.tolist() == [1.0]


def test_partial_action_allows_initial_l24_refresh():
    validate_observed_action('L8', 'L24', had_memory=False)


def test_partial_action_rejects_l24_when_memory_existed():
    with pytest.raises(RuntimeError, match='requested L8, observed L24'):
        validate_observed_action('L8', 'L24', had_memory=True)


def test_reuse_model_does_not_require_projector():
    model = SimpleNamespace(
        temporal_memory_controller=SimpleNamespace(updater_name='reuse'),
        pts_bbox_head=object(),
        temporal_memory_projector=None,
    )
    validate_profile_model(model)


def test_reuse_non_keyframe_has_canonical_action_name():
    controller = SimpleNamespace(updater_name='reuse', last_action='middle')
    assert reported_action(controller) == 'reuse'
    controller.last_action = 'L24'
    assert reported_action(controller) == 'L24'


def test_dense_model_is_valid_without_temporal_controller():
    model = SimpleNamespace(
        temporal_memory_controller=None,
        pts_bbox_head=object(),
    )
    validate_profile_model(model)


def test_dense_model_cannot_force_partial_action():
    model = SimpleNamespace(
        temporal_memory_controller=None,
        pts_bbox_head=object(),
    )
    with pytest.raises(ValueError, match='cannot force an action'):
        validate_profile_model(model, requested_action='L8')


def test_highres_profile_summary_contains_peak_memory_and_component_times():
    from tools.profile_streampetr_highres import build_summary

    summary = build_summary(
        resolution=(640, 1600),
        peak_allocated=40,
        peak_reserved=48,
        backbone_s=0.10,
        head_s=0.02,
        total_s=0.13,
    )
    assert summary == {
        'resolution': [640, 1600],
        'peak_allocated_bytes': 40,
        'peak_reserved_bytes': 48,
        'backbone_ms': 100.0,
        'head_ms': 20.0,
        'total_ms': 130.0,
    }


def test_highres_profile_requires_cuda():
    from tools.profile_streampetr_highres import ensure_cuda_available

    with pytest.raises(
        RuntimeError, match='CUDA is required for high-resolution profiling'):
        ensure_cuda_available('cpu', cuda_available=False)


def test_highres_profile_rejects_mixed_resolutions():
    from tools.profile_streampetr_highres import validate_consistent_resolution

    assert validate_consistent_resolution(None, [640, 1600]) == [640, 1600]
    assert validate_consistent_resolution([640, 1600], [640, 1600]) == [640, 1600]
    with pytest.raises(
        ValueError, match='expected all profiled samples to share resolution'):
        validate_consistent_resolution([640, 1600], [704, 1600])
