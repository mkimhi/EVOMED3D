import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.policy_feature_similarity import (
    camera_cka_values,
    camera_cosine_means,
    flatten_feature,
    layer_similarity_record,
    write_similarity_csv,
)


def test_camera_cosine_means_reports_one_value_per_camera():
    previous = torch.ones(1, 2, 3, 2, 2)
    current = previous.clone()
    current[:, 1] = -1.0

    values = camera_cosine_means(previous, current)

    assert values == pytest.approx([1.0, -1.0])


def test_layer_similarity_record_names_layer_camera_columns():
    previous = [torch.ones(1, 2, 3, 2, 2)]
    current = [torch.ones(1, 2, 3, 2, 2)]

    record = layer_similarity_record(
        previous,
        current,
        layer_name="L8",
        camera_names=["CAM_A", "CAM_B"],
    )

    assert record["similarity_L8_CAM_A"] == pytest.approx(1.0)
    assert record["similarity_L8_CAM_B"] == pytest.approx(1.0)
    assert record["similarity_L8_mean"] == pytest.approx(1.0)


def test_camera_cka_values_reports_one_value_per_camera():
    previous = torch.arange(1 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
    current = previous.clone()
    current[:, 1] = torch.flip(current[:, 1], dims=[-1])

    values = camera_cka_values(previous, current)

    assert values[0] == pytest.approx(1.0)
    assert len(values) == 2


def test_layer_similarity_record_can_include_patch_cka_columns():
    previous = [torch.arange(1 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 2, 3, 2, 2)]
    current = [previous[0].clone()]
    current[0][:, 1] = torch.flip(current[0][:, 1], dims=[-1])

    record = layer_similarity_record(
        previous,
        current,
        layer_name="L24",
        camera_names=["CAM_A", "CAM_B"],
        include_cka=True,
    )

    assert record["cka_L24_CAM_A"] == pytest.approx(1.0)
    assert "cka_L24_CAM_B" in record
    assert record["cka_L24_mean"] == pytest.approx(
        (record["cka_L24_CAM_A"] + record["cka_L24_CAM_B"]) / 2.0)
    assert record["cka_L24_min"] == pytest.approx(min(record["cka_L24_CAM_A"], record["cka_L24_CAM_B"]))
    assert record["cka_L24_max"] == pytest.approx(max(record["cka_L24_CAM_A"], record["cka_L24_CAM_B"]))
    assert record["cka_L24_spread"] == pytest.approx(record["cka_L24_max"] - record["cka_L24_min"])


def test_flatten_feature_accepts_singleton_list_only():
    feature = torch.ones(1, 2, 3, 4, 5)

    assert flatten_feature([feature]) is feature

    with pytest.raises(ValueError):
        flatten_feature([feature, feature])


def test_write_similarity_csv_includes_union_of_dynamic_columns(tmp_path):
    path = tmp_path / "similarity.csv"
    write_similarity_csv([
        {"token": "a", "similarity_L8_CAM_A": 1.0},
        {"token": "b", "similarity_L12_CAM_A": 0.5},
    ], path)

    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["similarity_L8_CAM_A"] == "1.0"
    assert rows[1]["similarity_L12_CAM_A"] == "0.5"


def test_normalize_queue_image_data_adds_batch_dim_for_scattered_queue():
    from tools.policy_feature_similarity import normalize_queue_image_data

    class Adapter:
        def _img(self, data):
            return data["img"][0]

    data, img = normalize_queue_image_data(
        Adapter(), {"img": [torch.zeros(2, 6, 3, 8, 8)]})

    assert tuple(img.shape) == (1, 2, 6, 3, 8, 8)
    assert tuple(data["img"][0].shape) == (1, 2, 6, 3, 8, 8)


def test_apply_dataset_info_fallback_fills_token_scene_and_frame():
    from tools.policy_feature_similarity import apply_dataset_info_fallback

    class Dataset:
        data_infos = [
            {"token": "tok0", "scene_token": "scene", "frame_idx": 7},
        ]

    records = apply_dataset_info_fallback(
        [{"token": "", "scene_token": "", "frame_number": 0}],
        Dataset(),
        start_index=0,
    )

    assert records[0]["token"] == "tok0"
    assert records[0]["scene_token"] == "scene"
    assert records[0]["frame_number"] == 7
