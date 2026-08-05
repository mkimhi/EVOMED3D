import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


MODULE_PATH = Path(__file__).parents[1] / 'tools' / 'train_temporal_policy_mlp.py'
SPEC = importlib.util.spec_from_file_location('train_temporal_policy_mlp', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

POLICY_PATH = (Path(__file__).parents[1] / 'projects' / 'mmdet3d_plugin'
               / 'models' / 'temporal_memory' / 'policies.py')
POLICY_SPEC = importlib.util.spec_from_file_location('temporal_policies', POLICY_PATH)
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY_MODULE)
MlpFramePolicy = POLICY_MODULE.MlpFramePolicy


def test_counterfactual_training_spec_uses_four_actions(tmp_path):
    csv_path = tmp_path / 'targets.csv'
    pd.DataFrame([
        dict(token='a', scene_token='s1', frame_number=0.0,
             num_annotations=1.0, error_L8=0.2,
             degradation_L8=0.1, oracle_action='L8'),
        dict(token='b', scene_token='s2', frame_number=1.0,
             num_annotations=2.0, error_L8=0.3,
             degradation_L8=0.2, oracle_action='L24'),
    ]).to_csv(csv_path, index=False)

    frame, features, y, labels, actions = MODULE.load_counterfactual_table(
        csv_path)

    assert labels == ['L8', 'L12', 'L16', 'L24']
    assert actions == {label: label for label in labels}
    assert features == ['frame_number', 'num_annotations']
    assert y.tolist() == [0, 3]
    assert len(frame) == 2


def test_counterfactual_export_loads_in_mlp_policy(tmp_path):
    manifest_path = tmp_path / 'targets_manifest.json'
    manifest_path.write_text(json.dumps({
        'teacher_forcing_residual_std': {
            'frame_number': 0.0,
            'num_annotations': 2.5,
        }
    }))
    model = MODULE.PolicyMlp(2, hidden_dim=4, out_dim=4)
    payload = MODULE.build_export_payload(
        model=model,
        features=['frame_number', 'num_annotations'],
        mean=[0.0, 1.0],
        scale=[1.0, 2.0],
        hidden_dim=4,
        labels=['L8', 'L12', 'L16', 'L24'],
        actions={label: label for label in ('L8', 'L12', 'L16', 'L24')},
        target='oracle_action',
        target_manifest=manifest_path,
        cv_summary={},
        class_counts={'L8': 1, 'L24': 1},
    )
    checkpoint = tmp_path / 'policy.pt'
    torch.save(payload, checkpoint)

    policy = MlpFramePolicy(checkpoint=str(checkpoint))

    assert policy.labels == ['L8', 'L12', 'L16', 'L24']
    assert policy.teacher_forcing_residual_std.tolist() == [0.0, 2.5]
