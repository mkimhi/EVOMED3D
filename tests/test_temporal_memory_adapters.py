import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_repdetr3d_adapter_extracts_full_features_with_existing_detector_path():
    from projects.mmdet3d_plugin.models.temporal_memory import build_detector_adapter

    features = [torch.randn(2, 6, 4, 3, 5), torch.randn(2, 6, 4, 2, 3)]

    class Model:
        def extract_feat(self, img, T, training_mode=False):
            self.extract_args = (img, T, training_mode)
            return features

    img = torch.randn(2, 1, 6, 3, 32, 32)
    model = Model()
    adapter = build_detector_adapter("repdetr3d")

    out = adapter.extract_full_features(model, {"img": img}, len_queue=1)

    assert out is features
    assert model.extract_args == (img, 1, False)


def test_repdetr3d_adapter_injects_features_into_head_without_mutating_batch():
    from projects.mmdet3d_plugin.models.temporal_memory import RepDetr3DAdapter

    features = [torch.randn(2, 6, 4, 3, 5)]
    img_metas = [{"scene_token": "a"}, {"scene_token": "b"}]

    class Head:
        def __call__(self, metas, **data):
            self.metas = metas
            self.data = data
            return {"ok": True}

    class Model:
        pts_bbox_head = Head()

    batch = {"img_feats": ["old"], "timestamp": torch.ones(2)}
    out = RepDetr3DAdapter().inject_features(Model(), batch, features, img_metas=img_metas)

    assert out == {"ok": True}
    assert Model.pts_bbox_head.metas is img_metas
    assert Model.pts_bbox_head.data["img_feats"] is features
    assert Model.pts_bbox_head.data["timestamp"] is batch["timestamp"]
    assert batch["img_feats"] == ["old"]


def test_repdetr3d_adapter_memory_packet_round_trip_preserves_feature_identity():
    from projects.mmdet3d_plugin.models.temporal_memory import RepDetr3DAdapter

    features = [torch.randn(2, 6, 4, 3, 5), torch.randn(2, 6, 4, 2, 3)]
    adapter = RepDetr3DAdapter()

    packet = adapter.feature_to_memory(features)
    restored = adapter.memory_to_feature(packet)

    assert packet.format == "spatial"
    assert packet.pre_pe is True
    assert packet.meta["pe_mode"] == "separate_recompute"
    assert packet.meta["pe_source"] == "head_or_transformer"
    assert packet.shape_meta == [(2, 6, 4, 3, 5), (2, 6, 4, 2, 3)]
    assert restored is features


def test_repdetr3d_adapter_asserts_feature_shape_when_debug_enabled():
    from projects.mmdet3d_plugin.models.temporal_memory import RepDetr3DAdapter

    with pytest.raises(AssertionError, match="5D"):
        RepDetr3DAdapter(debug_shapes=True).feature_to_memory([torch.randn(2, 3, 4)])


def test_streampetr_adapter_is_registered_but_refuses_unverified_wiring():
    from projects.mmdet3d_plugin.models.temporal_memory import StreamPETRAdapter, build_detector_adapter

    adapter = build_detector_adapter("streampetr")

    assert isinstance(adapter, StreamPETRAdapter)
    with pytest.raises(NotImplementedError, match="Petr3D.extract_img_feat"):
        adapter.extract_full_features(object(), {"img": torch.randn(1, 6, 3, 32, 32)})
    with pytest.raises(NotImplementedError, match="StreamPETRHead.position_embeding"):
        adapter.compute_position_encoding(object(), {}, None)
    with pytest.raises(NotImplementedError, match="Petr3D.prepare_location"):
        adapter.inject_features(object(), {}, torch.randn(1, 6, 4, 3, 5))
