import csv

import pytest

from projects.mmdet3d_plugin.models.temporal_memory.policy_targets import (
    CounterfactualActionTargets,
)


def write_targets(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['token', 'oracle_action'])
        writer.writeheader()
        for token, action in rows:
            writer.writerow({'token': token, 'oracle_action': action})
    return path


def test_counterfactual_target_provider_rejects_duplicates(tmp_path):
    path = write_targets(
        tmp_path / 'targets.csv', [('token', 'L8'), ('token', 'L24')])
    with pytest.raises(ValueError, match='duplicate'):
        CounterfactualActionTargets(path)


def test_counterfactual_target_provider_returns_label_index(tmp_path):
    path = write_targets(tmp_path / 'targets.csv', [('a', 'L16')])
    provider = CounterfactualActionTargets(path)
    assert provider.label_index('a', ['L8', 'L12', 'L16', 'L24']) == 2


def test_counterfactual_target_provider_reports_missing_token(tmp_path):
    path = write_targets(tmp_path / 'targets.csv', [('a', 'L16')])
    provider = CounterfactualActionTargets(path)
    with pytest.raises(KeyError, match='counterfactual policy target'):
        provider.label_index('missing', ['L8', 'L12', 'L16', 'L24'])


def test_counterfactual_target_provider_falls_back_to_scene_frame(tmp_path):
    path = tmp_path / 'targets.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=['token', 'scene_token', 'frame_number', 'oracle_action'])
        writer.writeheader()
        writer.writerow(dict(
            token='sample-a', scene_token='scene-a', frame_number=3,
            oracle_action='L24'))

    provider = CounterfactualActionTargets(path)

    assert provider.label_index(
        'unavailable-token', ['L8', 'L12', 'L16', 'L24'],
        scene_token='scene-a', frame_number=3) == 3
