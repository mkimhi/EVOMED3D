import numpy as np
import pytest
import subprocess
import sys
import torch

from tools.export_inference_examples import (
    activation_to_uint8_maps,
    artifact_stem,
    ordered_camera_paths,
    save_six_panel,
    select_scene_indices,
    validate_export_counts,
)


def test_select_scene_indices_takes_first_frames_from_distinct_scenes():
    infos = [
        {'scene_token': scene}
        for scene in ['scene-a', 'scene-a', 'scene-a', 'scene-b', 'scene-b',
                      'scene-c', 'scene-c']
    ]

    assert select_scene_indices(infos, num_scenes=3, frames_per_scene=2) == [
        (0, 0, 0),
        (1, 0, 1),
        (3, 1, 0),
        (4, 1, 1),
        (5, 2, 0),
        (6, 2, 1),
    ]


def test_activation_to_uint8_maps_keeps_all_six_cameras():
    feature = torch.arange(6 * 2 * 3 * 4).reshape(6, 2, 3, 4).float()

    maps = activation_to_uint8_maps(feature)

    assert len(maps) == 6
    assert all(item.shape == (2, 3) for item in maps)
    assert all(item.dtype == np.uint8 for item in maps)


def test_save_six_panel_writes_two_by_three_mosaic(tmp_path):
    images = [np.full((20, 30, 3), value, dtype=np.uint8)
              for value in range(0, 240, 40)]
    output_path = tmp_path / 'six_cameras.png'

    save_six_panel(images, [f'CAM_{index}' for index in range(6)], output_path)

    from PIL import Image
    with Image.open(output_path) as mosaic:
        assert mosaic.size == (960, 360)


def test_validate_export_counts_rejects_nonpositive_counts():
    with pytest.raises(ValueError, match='num-scenes'):
        validate_export_counts(num_scenes=0, frames_per_scene=5)
    with pytest.raises(ValueError, match='frames-per-scene'):
        validate_export_counts(num_scenes=3, frames_per_scene=0)


def test_artifact_stem_is_zero_padded():
    assert artifact_stem(scene_ordinal=2, frame_ordinal=4) == 'scene02_frame04'


def test_ordered_camera_paths_uses_canonical_six_camera_order(tmp_path):
    cameras = {
        name: {'data_path': f'{name}.jpg'}
        for name in reversed([
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
        ])
    }

    paths = ordered_camera_paths({'cams': cameras}, tmp_path)

    assert [path.name for _, path in paths] == [
        'CAM_FRONT_LEFT.jpg', 'CAM_FRONT.jpg', 'CAM_FRONT_RIGHT.jpg',
        'CAM_BACK_LEFT.jpg', 'CAM_BACK.jpg', 'CAM_BACK_RIGHT.jpg',
    ]


def test_ordered_camera_paths_does_not_duplicate_existing_data_root(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    camera_dir = tmp_path / 'data' / 'nuscenes'
    camera_dir.mkdir(parents=True)
    cameras = {}
    for name in [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']:
        relative_path = f'data/nuscenes/{name}.jpg'
        (tmp_path / relative_path).touch()
        cameras[name] = {'data_path': relative_path}

    paths = ordered_camera_paths({'cams': cameras}, camera_dir)

    assert paths[0][1] == tmp_path / 'data/nuscenes/CAM_FRONT_LEFT.jpg'


def test_cli_help_documents_export_controls():
    result = subprocess.run(
        [sys.executable, 'tools/export_inference_examples.py', '--help'],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '--num-scenes' in result.stdout
    assert '--frames-per-scene' in result.stdout
    assert '--feature-layer' in result.stdout
