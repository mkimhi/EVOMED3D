import sys
from pathlib import Path
from types import MethodType

import pytest
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_fake_petr3d(temporal_memory):
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
    from projects.mmdet3d_plugin.models.temporal_memory import (
        StreamPETRAdapter,
        TemporalMemoryController,
    )

    model = Petr3D.__new__(Petr3D)
    torch.nn.Module.__init__(model)
    model.temporal_memory = dict(temporal_memory)
    model.temporal_memory_enabled = bool(temporal_memory.get("enabled", False))
    model.tm_assert_no_teacher_in_inference = True
    model.full_calls = 0
    model.full_grad_enabled = []
    model.full_batch_values = []
    model.intermediate_layers = []
    model.intermediate_batch_values = []

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        assert len_queue == 1
        self.full_calls += 1
        self.full_grad_enabled.append(torch.is_grad_enabled())
        self.full_batch_values.append(img[:, 0, 0, 0, 0].tolist())
        return img + 100.0

    def extract_img_feat_intermediate(self, img, layer_id, len_queue=1):
        assert len_queue == 1
        self.intermediate_layers.append(layer_id)
        self.intermediate_batch_values.append(
            img[:, 0, 0, 0, 0].tolist())
        return img + float(layer_id)

    model.extract_img_feat = MethodType(extract_img_feat, model)
    model.extract_img_feat_intermediate = MethodType(
        extract_img_feat_intermediate, model)

    model.temporal_memory_adapter = None
    model.temporal_memory_controller = None
    model.temporal_memory_updater = None
    model.temporal_memory_projector = None
    if model.temporal_memory_enabled:
        updater = temporal_memory.get("updater", "gated_residual")
        updater_cfg = dict(temporal_memory.get("updater_cfg", {}))
        if updater == "gated_residual":
            updater_cfg.setdefault("channels", 1)
            updater_cfg.setdefault("residual_scale_init", 0.1)
        model.temporal_memory_adapter = StreamPETRAdapter()
        model.temporal_memory_controller = TemporalMemoryController(
            enabled=True,
            keyframe_interval=temporal_memory.get("keyframe_interval", 1),
            updater=updater,
            updater_cfg=updater_cfg,
            partial_layer=temporal_memory.get("partial_layer", "block12"),
            train_with_teacher=temporal_memory.get("train_with_teacher", False),
            distill_weight=temporal_memory.get("distill_weight", 1.0),
            inference_teacher_forbidden=True,
        )
        model.temporal_memory_updater = model.temporal_memory_controller.updater
    model.train(False)
    return model


def _frame(value=1.0, requires_grad=False):
    return torch.full((1, 2, 1, 1, 1), value, requires_grad=requires_grad)


def _metas(scene="scene-a", sample="sample-0"):
    return [{"scene_token": scene, "sample_idx": sample}]


def test_streampetr_k1_matches_full_extraction():
    model = make_fake_petr3d(
        temporal_memory=dict(enabled=True, keyframe_interval=1))
    img = _frame()

    features, losses = model._extract_img_feat_temporal_memory(
        img, _metas(), training=False)

    assert torch.equal(features, img + 100.0)
    assert losses == {}
    assert model.intermediate_layers == []


def test_streampetr_non_keyframe_uses_l12_and_never_calls_teacher_inference():
    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True,
        keyframe_interval=4,
        partial_layer="block12",
    ))

    model._extract_img_feat_temporal_memory(
        _frame(1.0), _metas(sample="sample-0"), training=False)
    model._extract_img_feat_temporal_memory(
        _frame(2.0), _metas(sample="sample-1"), training=False)

    assert model.intermediate_layers == [12]
    assert model.full_calls == 1


def test_streampetr_mixed_batch_extracts_only_each_routed_sample_subset():
    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True,
        keyframe_interval=4,
        partial_layer="block12",
        updater="partial_only",
    ))
    metas = [
        {"scene_token": "scene-a", "sample_idx": "a"},
        {"scene_token": "scene-b", "sample_idx": "b"},
    ]
    model._extract_img_feat_temporal_memory(
        torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1, 1),
        metas,
        training=False,
    )
    model.temporal_memory_controller.states[("scene-a", 0)]["frame_index"] = 4
    model.temporal_memory_controller.states[("scene-b", 1)]["frame_index"] = 1
    model.full_batch_values.clear()
    model.intermediate_batch_values.clear()

    features, _ = model._extract_img_feat_temporal_memory(
        torch.tensor([10.0, 20.0]).reshape(2, 1, 1, 1, 1),
        metas,
        training=False,
    )

    assert model.full_batch_values == [[10.0]]
    assert model.intermediate_batch_values == [[20.0]]
    assert features.reshape(-1).tolist() == [110.0, 32.0]


def test_streampetr_scene_boundary_forces_a_new_full_feature():
    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True,
        keyframe_interval=4,
        partial_layer="block12",
    ))

    model._extract_img_feat_temporal_memory(
        _frame(1.0), _metas(scene="scene-a", sample="a-0"), training=False)
    model._extract_img_feat_temporal_memory(
        _frame(2.0), _metas(scene="scene-a", sample="a-1"), training=False)
    features, losses = model._extract_img_feat_temporal_memory(
        _frame(3.0), _metas(scene="scene-b", sample="b-0"), training=False)

    assert torch.equal(features, _frame(103.0))
    assert losses == {}
    assert model.full_calls == 2
    assert model.intermediate_layers == [12]
    assert list(model.temporal_memory_controller.states) == [("scene-b", 0)]


def test_streampetr_disabled_preserves_direct_full_path():
    model = make_fake_petr3d(temporal_memory=dict(enabled=False))
    img = _frame(7.0)

    features, losses = model._extract_img_feat_temporal_memory(
        img, _metas(), training=False)

    assert torch.equal(features, img + 100.0)
    assert losses == {}
    assert model.temporal_memory_controller is None
    assert model.temporal_memory_adapter is None


def test_streampetr_disabled_training_helper_normalizes_queue_axis_to_5d():
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D

    model = Petr3D.__new__(Petr3D)
    torch.nn.Module.__init__(model)
    model.temporal_memory_enabled = False

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        assert training_mode is True
        return img.unsqueeze(1)

    model.extract_img_feat = MethodType(extract_img_feat, model)
    img = _frame(7.0)

    features, losses = model._extract_img_feat_temporal_memory(
        img, _metas(), training=True)

    assert features.shape == img.shape
    assert torch.equal(features, img)
    assert losses == {}


def test_streampetr_training_non_keyframe_distills_from_stop_gradient_teacher():
    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True,
        keyframe_interval=4,
        partial_layer="block12",
        train_with_teacher=True,
    ))
    teacher_requires_grad = []

    def recording_distillation(student, teacher):
        teacher_requires_grad.append(teacher.requires_grad)
        return F.smooth_l1_loss(student, teacher)

    model.temporal_memory_controller.distillation_loss = recording_distillation
    model.train(True)
    model._extract_img_feat_temporal_memory(
        _frame(1.0, requires_grad=True),
        _metas(sample="sample-0"),
        training=True,
    )
    _, losses = model._extract_img_feat_temporal_memory(
        _frame(2.0, requires_grad=True),
        _metas(sample="sample-1"),
        training=True,
    )

    assert list(losses) == ["loss_tm_distill"]
    assert losses["loss_tm_distill"].requires_grad
    assert teacher_requires_grad == [False]
    assert model.full_grad_enabled[-1] is False


def test_streampetr_inference_rejects_a_supplied_non_keyframe_teacher():
    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True,
        keyframe_interval=4,
        partial_layer="block12",
    ))
    model._extract_img_feat_temporal_memory(
        _frame(1.0), _metas(sample="sample-0"), training=False)

    with pytest.raises(
        AssertionError,
        match="Full-feature teacher is forbidden on non-keyframe inference",
    ):
        model._extract_img_feat_temporal_memory(
            _frame(2.0),
            _metas(sample="sample-1"),
            training=False,
            teacher_features=_frame(102.0),
        )


def test_streampetr_inference_teacher_guard_uses_resolved_controller_config():
    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True,
        keyframe_interval=4,
        partial_layer="block12",
    ))
    model.tm_assert_no_teacher_in_inference = True
    model.temporal_memory_controller.inference_teacher_forbidden = False
    model._extract_img_feat_temporal_memory(
        _frame(1.0), _metas(sample="sample-0"), training=False)

    features, losses = model._extract_img_feat_temporal_memory(
        _frame(2.0),
        _metas(sample="sample-1"),
        training=False,
        teacher_features=_frame(102.0),
    )

    assert features.shape == _frame().shape
    assert losses == {}


def test_streampetr_forward_train_uses_one_router_and_collects_distillation():
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D

    model = Petr3D.__new__(Petr3D)
    torch.nn.Module.__init__(model)
    model.temporal_memory_enabled = True
    model.test_flag = False
    model.num_frame_backbone_grads = 1
    model.num_frame_head_grads = 1
    model.num_frame_losses = 1
    routed = []
    seen_features = []

    def route(self, img, img_metas, training=False, prev_exists=None):
        routed.append((img.clone(), img_metas, training, prev_exists.clone()))
        frame = len(routed)
        losses = {"loss_tm_distill": img.new_tensor(
            float(frame), requires_grad=torch.is_grad_enabled())}
        return img[:, :, :1, :1, :1] + frame, losses

    def history(self, *args, **data):
        seen_features.append(data["img_feats"])
        return {"loss_detector": data["img_feats"].sum()}

    model._extract_img_feat_temporal_memory = MethodType(route, model)
    model.obtain_history_memory = MethodType(history, model)
    img = torch.zeros(1, 2, 2, 3, 1, 1)
    metas = [[{"scene_token": "scene-a"}], [{"scene_token": "scene-a"}]]

    losses = model.forward_train(
        img_metas=metas,
        gt_bboxes_3d=[None, None],
        gt_labels_3d=[None, None],
        gt_bboxes=[None, None],
        gt_labels=[None, None],
        centers2d=[None, None],
        depths=[None, None],
        prev_exists=torch.tensor([[0.0, 1.0]]),
        img=img,
    )

    assert len(routed) == 2
    assert all(call[2] is True for call in routed)
    assert [call[3].item() for call in routed] == [0.0, 1.0]
    assert seen_features[0].shape == (1, 2, 2, 1, 1, 1)
    assert losses["loss_tm_distill"].requires_grad
    assert losses["loss_tm_distill"].item() == pytest.approx(2.0)


def test_streampetr_simple_test_routes_features_before_unchanged_head_path():
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D

    model = Petr3D.__new__(Petr3D)
    torch.nn.Module.__init__(model)
    model.temporal_memory_enabled = True
    seen = {}

    def route(self, img, img_metas, training=False, prev_exists=None):
        seen["route"] = (img, img_metas, training, prev_exists)
        return img + 9.0, {}

    def simple_test_pts(self, img_metas, **data):
        seen["head"] = (img_metas, data["img_feats"])
        return ["boxes"]

    model._extract_img_feat_temporal_memory = MethodType(route, model)
    model.simple_test_pts = MethodType(simple_test_pts, model)
    img = _frame(4.0)
    metas = _metas()

    results = model.simple_test(metas, img=img)

    assert seen["route"] == (img, metas, False, None)
    assert seen["head"][0] is metas
    assert torch.equal(seen["head"][1], img + 9.0)
    assert results == [{"pts_bbox": "boxes"}]
