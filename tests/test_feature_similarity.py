import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.feature_similarity import (
    RepDETR3DFeatureAnalysisAdapter,
    StreamPETRFeatureAnalysisAdapter,
    compare_feature_sets,
    configure_frame_gap_dataset_cfg,
    cuda_target_gpus,
    limit_dataset_by_scenes,
    linear_cka,
    pearsonr,
    save_camera_heatmaps,
    single_model_mode,
    spearmanr,
    temporal_feature_pair,
)


def test_compare_feature_sets_reports_global_per_camera_and_spatial_metrics():
    teacher = [torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 2, 2) + 1.0]
    student = [teacher[0].clone()]
    foreground = torch.tensor([[[[True, False], [False, False]], [[False, True], [False, False]]]])

    metrics, heatmaps = compare_feature_sets(
        student,
        teacher,
        camera_names=["CAM_A", "CAM_B"],
        masks={"foreground": foreground},
    )

    assert metrics["cosine/global_mean"] == pytest.approx(1.0)
    assert metrics["cosine/CAM_A_mean"] == pytest.approx(1.0)
    assert metrics["cosine/CAM_B_mean"] == pytest.approx(1.0)
    assert metrics["cosine/foreground_mean"] == pytest.approx(1.0)
    assert metrics["cosine/background_mean"] == pytest.approx(1.0)
    assert metrics["cka/center"] == pytest.approx(1.0)
    assert metrics["cka/border"] == pytest.approx(1.0)
    assert heatmaps["level0"].shape == (2, 2, 2)


def test_linear_cka_and_rsm_correlations_are_one_for_identical_features():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = x.clone()
    values = torch.tensor([3.0, 1.0, 2.0])

    assert torch.isclose(linear_cka(x, y), torch.tensor(1.0))
    assert pearsonr(values, values) == 1.0
    assert spearmanr(values, values) == 1.0


def test_save_camera_heatmaps_writes_png_and_summary(tmp_path):
    heatmaps = {"level0": np.ones((2, 3, 4), dtype=np.float32)}
    summary = {"cosine/global_mean": 1.0}

    save_camera_heatmaps(
        heatmaps,
        tmp_path,
        camera_names=["CAM_A", "CAM_B"],
        summary=summary,
    )

    assert (tmp_path / "level0_CAM_A.png").exists()
    assert (tmp_path / "level0_CAM_B.png").exists()
    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text())["cosine/global_mean"] == 1.0


def test_repdetr3d_adapter_builds_foreground_and_depth_masks():
    adapter = RepDETR3DFeatureAnalysisAdapter()
    data = {
        "img": torch.zeros(1, 2, 3, 40, 80),
        "gt_bboxes": [[
            torch.tensor([[0.0, 0.0, 40.0, 20.0]]),
            torch.tensor([[40.0, 20.0, 80.0, 40.0]]),
        ]],
        "depths": [[
            torch.tensor([10.0]),
            torch.tensor([30.0]),
        ]],
    }

    masks = adapter.masks(data, (1, 2, 4, 4, 8))

    assert masks["foreground"].shape == (1, 2, 4, 8)
    assert masks["foreground"].any()
    assert masks["near_foreground"].any()
    assert masks["far_foreground"].any()


def test_streampetr_adapter_wraps_full_features_as_single_level():
    class FakeStreamPETR:
        def extract_img_feat(self, img, len_queue=1, training_mode=False):
            assert len_queue == 1
            assert training_mode
            return torch.ones(1, 1, 2, 3, 4, 5)

    adapter = StreamPETRFeatureAnalysisAdapter()
    features = adapter.extract_teacher_features(FakeStreamPETR(), {"img": torch.zeros(1, 2, 3, 8, 8)})

    assert len(features) == 1
    assert features[0].shape == (1, 2, 3, 4, 5)


def test_repdetr3d_sequence_extraction_restores_batch_dim_from_scattered_queue():
    class FakeModel:
        def __init__(self):
            self.seen_shape = None

        def extract_img_feat(self, img, len_queue=1, training_mode=False):
            self.seen_shape = tuple(img.shape)
            assert len_queue == 5
            assert training_mode
            return [torch.ones(1, 5, 6, 2, 3, 4)]

    model = FakeModel()
    adapter = RepDETR3DFeatureAnalysisAdapter()
    features = adapter.extract_final_sequence_features(
        model,
        {"img": [torch.zeros(5, 6, 3, 8, 8)]},
        queue_length=5,
    )

    assert model.seen_shape == (1, 5, 6, 3, 8, 8)
    assert features[0].shape == (1, 5, 6, 2, 3, 4)


def test_repdetr3d_adapter_extracts_raw_backbone_layer_features():
    class FakeBackbone:
        def __init__(self):
            self.layer_indices = []

        def forward_intermediate(self, img, layer_index):
            self.layer_indices.append(layer_index)
            return torch.ones(img.size(0), 4, 2, 3) * layer_index

    class FakeModel:
        use_grid_mask = False

        def __init__(self):
            self.img_backbone = FakeBackbone()

    model = FakeModel()
    adapter = RepDETR3DFeatureAnalysisAdapter()
    features = adapter.extract_backbone_layer_features(
        model, {"img": torch.zeros(1, 2, 3, 16, 24)}, "block8")

    assert len(features) == 1
    assert features[0].shape == (1, 2, 4, 2, 3)
    assert features[0].mean().item() == pytest.approx(8.0)
    assert model.img_backbone.layer_indices == [8]


def test_repdetr3d_adapter_extracts_final_raw_backbone_layer():
    class FakeBackbone:
        def __init__(self):
            self.layer_indices = []

        def forward_intermediate(self, img, layer_index):
            self.layer_indices.append(layer_index)
            return torch.ones(img.size(0), 4, 2, 3) * layer_index

    class FakeModel:
        use_grid_mask = False

        def __init__(self):
            self.img_backbone = FakeBackbone()

    model = FakeModel()
    adapter = RepDETR3DFeatureAnalysisAdapter()
    features = adapter.extract_backbone_layer_features(
        model, {"img": torch.zeros(1, 2, 3, 16, 24)}, "final", depth=24)

    assert features[0].mean().item() == pytest.approx(23.0)
    assert model.img_backbone.layer_indices == [23]


def test_temporal_feature_pair_returns_anchor_and_future_frame():
    features = [torch.arange(1 * 5 * 2 * 1 * 1 * 1).reshape(1, 5, 2, 1, 1, 1).float()]

    anchor, future = temporal_feature_pair(features, gap=3)

    assert anchor[0].shape == (1, 2, 1, 1, 1)
    assert future[0].shape == (1, 2, 1, 1, 1)
    assert anchor[0].flatten().tolist() == [0.0, 1.0]
    assert future[0].flatten().tolist() == [6.0, 7.0]


def test_limit_dataset_by_scenes_keeps_first_unique_scene_tokens():
    class DummyDataset:
        def __init__(self):
            self.data_infos = [
                {"scene_token": "a", "token": "a0"},
                {"scene_token": "a", "token": "a1"},
                {"scene_token": "b", "token": "b0"},
                {"scene_token": "c", "token": "c0"},
            ]
            self.flag = np.arange(4)

        def __len__(self):
            return len(self.data_infos)

    dataset = DummyDataset()

    limit_dataset_by_scenes(dataset, 2)

    assert [info["token"] for info in dataset.data_infos] == ["a0", "a1", "b0"]
    assert dataset.flag.tolist() == [0, 1, 2]


def test_frame_gap_final_is_single_model_mode():
    assert single_model_mode("layer-vs-final")
    assert single_model_mode("frame-gap-final")
    assert not single_model_mode("two-model")


def test_cuda_target_gpus_returns_integer_ids_for_mmcv_scatter():
    assert cuda_target_gpus("cuda") == [0]
    assert cuda_target_gpus("cuda:0") == [0]
    assert cuda_target_gpus("cuda:3") == [3]


def test_configure_frame_gap_dataset_cfg_disables_seq_mode_queue_reset():
    cfg = {
        "queue_length": 1,
        "random_length": 3,
        "seq_mode": True,
    }

    configure_frame_gap_dataset_cfg(cfg, queue_length=5)

    assert cfg["queue_length"] == 5
    assert cfg["random_length"] == 0
    assert cfg["seq_mode"] is False
