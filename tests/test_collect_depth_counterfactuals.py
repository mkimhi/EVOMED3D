import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / 'tools' / 'collect_depth_counterfactuals.py'
SPEC = importlib.util.spec_from_file_location('collect_depth_counterfactuals', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ACTIONS = MODULE.ACTIONS
build_manifest = MODULE.build_manifest
capture_runtime_state = MODULE.capture_runtime_state
collect_frame_branches = MODULE.collect_frame_branches
forced_action = MODULE.forced_action
restore_runtime_state = MODULE.restore_runtime_state
validate_model = MODULE.validate_model
validate_output_dir = MODULE.validate_output_dir
validate_token_sets = MODULE.validate_token_sets


def fake_model():
    controller = SimpleNamespace(
        frame_index=2,
        frames_since_full=1,
        scene_token='scene-a',
        memory=[torch.tensor([1.0])],
        last_action='L12',
        last_action_forced=False,
        last_distillation_loss=None,
        _policy_stats={'L12': 1},
        policy=SimpleNamespace(name='deployed'),
        max_frames_between_full=4,
    )
    head = SimpleNamespace(
        memory_embedding=torch.tensor([2.0]),
        memory_reference_point=torch.tensor([3.0]),
        memory_timestamp=torch.tensor([4.0]),
        memory_egopose=torch.tensor([5.0]),
        memory_velo=torch.tensor([6.0]),
        latest_routing_metadata={'kept': torch.tensor([7.0])},
    )
    return SimpleNamespace(
        temporal_memory_controller=controller,
        temporal_memory_projector=SimpleNamespace(projectors={
            'L8': object(), 'L12': object(), 'L16': object()}),
        pts_bbox_head=head,
        prev_scene_token='scene-a',
        test_flag=True,
        _tm_policy_prev_state={'timestamp': 3},
        _tm_policy_previous_l8_similarity=0.75,
    )


def test_runtime_snapshot_clones_and_restores_state():
    model = fake_model()
    snapshot = capture_runtime_state(model)
    model.temporal_memory_controller.memory[0].add_(10)
    model.pts_bbox_head.memory_embedding.zero_()
    model._tm_policy_prev_state['timestamp'] = 99
    model._tm_policy_previous_l8_similarity = 0.10
    model._tm_policy_prev_extra = torch.tensor([8.0])

    assert torch.equal(snapshot['controller']['memory'][0], torch.tensor([1.0]))
    restore_runtime_state(model, snapshot)

    assert torch.equal(model.temporal_memory_controller.memory[0], torch.tensor([1.0]))
    assert torch.equal(model.pts_bbox_head.memory_embedding, torch.tensor([2.0]))
    assert model._tm_policy_prev_state['timestamp'] == 3
    assert model._tm_policy_previous_l8_similarity == 0.75
    assert not hasattr(model, '_tm_policy_prev_extra')


def test_forced_action_restores_policy_and_age_after_error():
    controller = fake_model().temporal_memory_controller
    original_policy = controller.policy
    with pytest.raises(RuntimeError):
        with forced_action(controller, 'L8'):
            assert controller.policy.decide(has_memory=True) == 'L8'
            assert controller.policy.decide(has_memory=False) == 'L24'
            assert controller.policy.decide(has_memory=True, is_reset_frame=True) == 'L24'
            assert controller.max_frames_between_full is None
            raise RuntimeError('branch failed')
    assert controller.policy is original_policy
    assert controller.max_frames_between_full == 4


def test_collect_frame_branches_share_pre_state_and_restore_canonical_post_state():
    model = fake_model()
    observed_pre_values = []

    def run_inference(forced):
        controller = model.temporal_memory_controller
        observed_pre_values.append(float(controller.memory[0].item()))
        action = 'L12' if forced is None else controller.policy.decide(has_memory=True)
        controller.last_action = action
        controller.memory[0].add_(ACTIONS.index(action) + 1)
        model.pts_bbox_head.memory_embedding.add_(1)
        return f'prediction-{action}'

    predictions, canonical_action = collect_frame_branches(model, run_inference)

    assert canonical_action == 'L12'
    assert set(predictions) == set(ACTIONS)
    assert observed_pre_values == [1.0, 1.0, 1.0, 1.0]
    assert torch.equal(model.temporal_memory_controller.memory[0], torch.tensor([3.0]))
    assert torch.equal(model.pts_bbox_head.memory_embedding, torch.tensor([3.0]))
    assert model.temporal_memory_controller.last_action == 'L12'


def test_validation_and_manifest_helpers(tmp_path):
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    validate_output_dir(output_dir)
    (output_dir / 'partial.txt').write_text('partial')
    with pytest.raises(FileExistsError):
        validate_output_dir(output_dir)

    with pytest.raises(ValueError):
        validate_model(SimpleNamespace())
    validate_model(fake_model())
    numeric_key_model = fake_model()
    numeric_key_model.temporal_memory_projector.projectors = {
        '8': object(), '12': object(), '16': object()}
    validate_model(numeric_key_model)

    token_sets = {action: {'a', 'b'} for action in ACTIONS}
    assert validate_token_sets(token_sets) == {'a', 'b'}
    token_sets['L12'] = {'a'}
    with pytest.raises(ValueError):
        validate_token_sets(token_sets)

    manifest = build_manifest(
        run_id='run-1',
        config_path=tmp_path / 'config.py',
        checkpoint_path=tmp_path / 'model.pth',
        prediction_paths={action: tmp_path / action for action in ACTIONS},
        tokens={'b', 'a'},
        action_counts={'L8': 1, 'L12': 1},
        cfg_options={'model.temporal_memory.max_frames_between_full': 4},
        config_sha256='config-hash',
        checkpoint_sha256='checkpoint-hash',
    )
    assert manifest['shared_pre_frame_state'] is True
    assert manifest['canonical_state_advance_mode'] == 'deployed_policy'
    assert manifest['sample_count'] == 2
    assert set(manifest['prediction_paths']) == set(ACTIONS)
    assert len(manifest['token_set_sha256']) == 64
    assert manifest['cfg_options'] == {
        'model.temporal_memory.max_frames_between_full': 4}


def test_prepare_test_dataset_cfg_removes_dataloader_only_keys():
    original = {'type': 'Dataset', 'test_mode': False, 'samples_per_gpu': 2}

    prepared = MODULE.prepare_test_dataset_cfg(original)

    assert prepared == {'type': 'Dataset', 'test_mode': True}
    assert original['samples_per_gpu'] == 2


def test_select_dataset_cfg_uses_requested_split():
    cfg = SimpleNamespace(data=SimpleNamespace(
        train={'type': 'Dataset', 'ann_file': 'train.pkl', 'samples_per_gpu': 2},
        val={'type': 'Dataset', 'ann_file': 'val.pkl'},
        test={'type': 'Dataset', 'ann_file': 'test.pkl'},
    ))

    selected = MODULE.select_dataset_cfg(cfg, 'train')

    assert selected['ann_file'] == 'train.pkl'
    assert selected['test_mode'] is True
    assert 'samples_per_gpu' not in selected
