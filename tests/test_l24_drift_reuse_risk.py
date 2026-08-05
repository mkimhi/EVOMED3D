import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.l24_drift_reuse_risk as reuse_risk

from tools.l24_drift_reuse_risk import (
    frame_error,
    full_feature_cosine,
    scene_bootstrap_summary,
    select_reuse_targets,
    validate_collection_manifest,
)
from tools.collect_l24_drift_reuse_risk import (
    _assert_reuse_action,
    batch_token_and_scene,
    order_target_predictions,
    build_collection_manifest,
    record_target_drift,
)


def box(name, xy, score):
    return {
        'name': name,
        'translation': np.array([xy[0], xy[1], 0.0]),
        'size': np.array([2.0, 4.0, 1.5]),
        'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
        'velocity': np.array([0.0, 0.0]),
        'score': score,
    }


def feature(value):
    return [torch.full((1, 1, 2, 1, 1), value)]


def test_select_reuse_targets_is_scene_balanced_and_skips_keyframes():
    infos = ([{'token': f'a-{i}', 'scene_token': 'a'} for i in range(50)] +
             [{'token': f'b-{i}', 'scene_token': 'b'} for i in range(50)])
    rows = select_reuse_targets(infos, num_scenes=2, targets_per_scene=20)
    assert len(rows) == 40
    assert {row['scene_token'] for row in rows} == {'a', 'b'}
    assert all(row['frame_index'] >= 2 and row['frame_index'] % 2 == 1 for row in rows)


def test_select_reuse_targets_rejects_duplicate_tokens_and_noncontiguous_scenes():
    duplicate = [{'token': 'same', 'scene_token': 'a'}] * 5
    with pytest.raises(ValueError, match='duplicate token'):
        select_reuse_targets(duplicate, num_scenes=1, targets_per_scene=1)

    repeated = [{'token': 'a-0', 'scene_token': 'a'}, {'token': 'b-0', 'scene_token': 'b'},
                {'token': 'a-1', 'scene_token': 'a'}]
    with pytest.raises(ValueError, match='non-contiguous'):
        select_reuse_targets(repeated, num_scenes=1, targets_per_scene=1)


def test_full_feature_cosine_handles_identity_and_orthogonal_inputs():
    x = [torch.tensor([[[[[1.0]], [[0.0]]]]])]
    y = [torch.tensor([[[[[0.0]], [[1.0]]]]])]
    assert full_feature_cosine(x, x) == pytest.approx(1.0)
    assert full_feature_cosine(x, y) == pytest.approx(0.0)


def test_frame_error_excludes_confidence_for_box_loss():
    target = box('car', (0, 0), score=1.0)
    prediction = box('car', (0, 0), score=0.25)
    assert frame_error([], [target], True) == pytest.approx(1.0)
    assert frame_error([prediction], [target], True) > 0.0
    assert frame_error([prediction], [target], False) == pytest.approx(0.0)


def test_score_floor_excludes_low_confidence_boxes_from_both_endpoints():
    target = box('car', (0, 0), score=1.0)
    above_floor = box('car', (0, 0), score=0.50)
    below_floor = box('car', (20, 20), score=0.09)
    filtered = reuse_risk.filter_predictions_by_score(
        [above_floor, below_floor], score_floor=0.10)
    assert filtered == [above_floor]
    assert frame_error(filtered, [target], True) == pytest.approx(
        frame_error([above_floor], [target], True))
    assert frame_error(filtered, [target], False) == pytest.approx(
        frame_error([above_floor], [target], False))


def test_full_feature_cosine_statistics_reconstruct_both_drift_offsets():
    current = [torch.tensor([[[[[1.0]], [[0.0]]]]])]
    previous_dt1 = [torch.tensor([[[[[0.0]], [[1.0]]]]])]
    previous_dt2 = [torch.tensor([[[[[1.0]], [[0.0]]]]])]
    dt1_statistics = reuse_risk.full_feature_cosine_statistics(previous_dt1, current)
    dt2_statistics = reuse_risk.full_feature_cosine_statistics(previous_dt2, current)
    assert reuse_risk.reconstruct_full_feature_cosine(dt1_statistics) == pytest.approx(0.0)
    assert reuse_risk.reconstruct_full_feature_cosine(dt2_statistics) == pytest.approx(1.0)
    assert dt1_statistics['levels'][0]['vector_count'] == 1


def test_zero_free_data_limits_are_finite_and_do_not_force_zero():
    lower, upper = reuse_risk.zero_free_data_limits([0.1, 0.5, 0.9])
    assert np.isfinite([lower, upper]).all()
    assert 0.0 < lower < upper


def test_scene_bootstrap_returns_five_quantiles_and_perfect_rho():
    frame = pd.DataFrame({'scene_token': np.repeat(['a', 'b', 'c'], 5),
                          'drift_dt1': np.arange(15.), 'prediction_loss': np.arange(15.)})
    summary, correlation = scene_bootstrap_summary(
        frame, 'drift_dt1', 'prediction_loss', 20, 7)
    assert len(summary) == 5
    assert correlation['spearman'] == pytest.approx(1.0)


def test_build_panel_summary_has_every_offset_outcome_pair():
    frame = pd.DataFrame({
        'scene_token': np.repeat(['a', 'b'], 10),
        'drift_dt1': np.arange(20.),
        'drift_dt2': np.arange(20.) + 1,
        'prediction_loss': np.arange(20.),
        'box_loss': np.arange(20.)[::-1],
    })
    _, correlations = reuse_risk.build_panel_summary(frame, repetitions=20, seed=4)
    assert set(zip(correlations.offset, correlations.outcome)) == {
        ('dt1', 'prediction_loss'), ('dt2', 'prediction_loss'),
        ('dt1', 'box_loss'), ('dt2', 'box_loss'),
    }


def _valid_target_tables():
    tokens = [f'scene-{scene:03d}-target-{target:02d}'
              for scene in range(100) for target in range(20)]
    scenes = [f'scene-{scene:03d}' for scene in range(100) for _ in range(20)]
    selection = pd.DataFrame({'token': tokens, 'scene_token': scenes})
    drift = selection.assign(drift_dt1=np.linspace(0.0, 1.0, len(selection)),
                             drift_dt2=np.linspace(0.1, 1.1, len(selection)))
    return selection, drift, {token: [] for token in tokens}


def test_validate_target_tables_requires_exactly_twenty_targets_per_scene():
    selection, drift, predictions = _valid_target_tables()
    selection.loc[0, 'scene_token'] = 'scene-001'
    with pytest.raises(ValueError, match='exactly 20 targets per scene'):
        reuse_risk.validate_target_tables(selection, drift, predictions)


def test_validate_target_tables_rejects_token_to_scene_mismatch_before_merge():
    selection, drift, predictions = _valid_target_tables()
    drift.loc[0, 'scene_token'] = 'wrong-scene'
    with pytest.raises(ValueError, match='token-to-scene mapping'):
        reuse_risk.validate_target_tables(selection, drift, predictions)


def test_validate_collection_manifest_resolves_required_artifact_paths(tmp_path):
    paths = {'selection': tmp_path / 'selection.csv', 'drift': tmp_path / 'drift.csv',
             'drift_statistics': tmp_path / 'drift_statistics.json',
             'predictions': tmp_path / 'results_nusc.json'}
    paths['selection'].write_text('token,scene_token,frame_index\na,scene-a,3\n')
    for name, path in paths.items():
        if name != 'selection':
            path.write_text('{}')
    manifest = {
        'representation_source': 'base_full_l24_backbone_neck',
        'prediction_mode': 'naive_reuse_k2',
        'keyframe_interval': 2,
        'selection_path': str(paths['selection']),
        'drift_path': str(paths['drift']),
        'drift_statistics_path': str(paths['drift_statistics']),
        'prediction_path': str(paths['predictions']),
        'dataset_version': 'v1.0-trainval',
        'selected_scene_tokens': ['scene-a'],
        'selected_target_tokens': ['a'],
        'selection_sha256': hashlib.sha256(paths['selection'].read_bytes()).hexdigest(),
        'qualified_scene_count': 1,
        'selected_target_count': 1,
    }
    assert validate_collection_manifest(manifest) == {
        'selection': paths['selection'], 'drift': paths['drift'],
        'drift_statistics': paths['drift_statistics'],
        'predictions': paths['predictions'],
    }


def test_validate_collection_manifest_rejects_selection_identity_mismatch(tmp_path):
    selection_path = tmp_path / 'selection.csv'
    selection_path.write_text('token,scene_token,frame_index\na,scene-a,3\nb,scene-a,5\n')
    paths = {
        'drift_path': tmp_path / 'drift.csv',
        'prediction_path': tmp_path / 'results_nusc.json',
        'drift_statistics_path': tmp_path / 'drift_statistics.json',
    }
    for path in paths.values():
        path.write_text('{}')
    manifest = {
        'representation_source': 'base_full_l24_backbone_neck',
        'prediction_mode': 'naive_reuse_k2',
        'keyframe_interval': 2,
        'selection_path': str(selection_path),
        **{name: str(path) for name, path in paths.items()},
        'dataset_version': 'v1.0-trainval',
        'selected_scene_tokens': ['scene-a'],
        'selected_target_tokens': ['b', 'a'],
        'selection_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        'qualified_scene_count': 1,
        'selected_target_count': 2,
    }
    with pytest.raises(ValueError, match='selected target tokens'):
        validate_collection_manifest(manifest)


def test_validate_collection_manifest_rejects_wrong_provenance(tmp_path):
    manifest = {
        'representation_source': 'partial_l8',
        'prediction_mode': 'naive_reuse_k2',
        'keyframe_interval': 2,
        'selection_path': str(tmp_path / 'selection.csv'),
        'drift_path': str(tmp_path / 'drift.csv'),
        'prediction_path': str(tmp_path / 'results_nusc.json'),
    }
    with pytest.raises(ValueError, match='base full L24'):
        validate_collection_manifest(manifest)


def test_manifest_marks_base_l24_and_naive_reuse_k2(tmp_path):
    selection_path = tmp_path / 'selection.csv'
    selection_path.write_text('token,scene_token,frame_index\na,scene-a,3\n')
    drift_statistics_path = tmp_path / 'drift_statistics.json'
    drift_statistics_path.write_text('{}')
    manifest = build_collection_manifest(
        'run', tmp_path / 'c.py', tmp_path / 'm.pth',
        selection_path, tmp_path / 'drift.csv',
        tmp_path / 'results.json', 'config-hash', 'checkpoint-hash',
        9, 100, 2000, drift_statistics_path=drift_statistics_path,
        dataset_version='v1.0-trainval', selected_scene_tokens=['scene-a'],
        selected_target_tokens=['a'])
    assert manifest['representation_source'] == 'base_full_l24_backbone_neck'
    assert manifest['prediction_mode'] == 'naive_reuse_k2'
    assert manifest['keyframe_interval'] == 2
    assert manifest['dataset_version'] == 'v1.0-trainval'
    assert manifest['selected_target_tokens'] == ['a']


def test_target_drift_rejects_cross_scene_or_short_history():
    with pytest.raises(ValueError, match='two full-L24 history frames'):
        record_target_drift('b-0', 'scene-b', {'scene-a': [feature(1.0)]})


def test_target_requires_the_configured_naive_reuse_action():
    controller = SimpleNamespace(
        updater_name='reuse', policy=SimpleNamespace(partial_action='Lmiddle'),
        last_action=None)
    model = SimpleNamespace(temporal_memory_controller=controller)
    with pytest.raises(ValueError, match='naive-reuse action'):
        _assert_reuse_action(model)
    controller.last_action = 'L8'
    with pytest.raises(ValueError, match='naive-reuse action'):
        _assert_reuse_action(model)
    controller.last_action = 'Lmiddle'
    _assert_reuse_action(model)


def test_target_predictions_are_ordered_by_selected_tokens():
    predictions = order_target_predictions(
        ['first', 'second'], {'second': ['prediction-2'], 'first': ['prediction-1']})
    assert predictions == ['prediction-1', 'prediction-2']


def test_batch_metadata_must_match_the_expected_frame_identity():
    assert batch_token_and_scene({
        'img_metas': [[{'sample_idx': 'sample-1', 'scene_token': 'scene-1'}]],
    }) == ('sample-1', 'scene-1')
    with pytest.raises(ValueError, match='token and scene_token'):
        batch_token_and_scene({'img_metas': [[{'sample_idx': 'sample-1'}]]})
