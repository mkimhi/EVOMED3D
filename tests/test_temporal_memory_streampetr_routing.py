import sys
import math
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


def test_streampetr_reuse_non_keyframe_never_extracts_partial_features():
    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True,
        keyframe_interval=4,
        partial_layer="block12",
        updater="reuse",
    ))

    model._extract_img_feat_temporal_memory(
        _frame(1.0), _metas(sample="sample-0"), training=False)
    features, _ = model._extract_img_feat_temporal_memory(
        _frame(2.0), _metas(sample="sample-1"), training=False)

    assert model.intermediate_layers == []
    assert torch.equal(features, _frame(101.0))


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


def test_streampetr_policy_receives_per_sample_features_and_is_registered():
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    class FeaturePolicy(torch.nn.Module):
        feature_names = ["desired_action"]

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen = []

        def decide(self, has_memory, is_reset_frame=False,
                   policy_features=None, **kwargs):
            self.seen.append(dict(policy_features or {}))
            if is_reset_frame or not has_memory:
                return "L24"
            return policy_features["desired_action"]

    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True, updater="partial_only", partial_layer="block12"))
    policy = FeaturePolicy()
    model.temporal_memory_controller = TemporalMemoryController(
        enabled=True, updater="partial_only", policy=policy)
    model.temporal_memory_policy = policy
    assert "temporal_memory_policy.weight" in dict(model.named_parameters())
    metas = [
        {"scene_token": "scene-a", "desired_action": "L8"},
        {"scene_token": "scene-b", "desired_action": "L12"},
    ]
    img = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1, 1)
    model._extract_img_feat_temporal_memory(img, metas, training=False)
    model._extract_img_feat_temporal_memory(img, metas, training=False)

    assert policy.seen[-2]["desired_action"] == "L8"
    assert policy.seen[-1]["desired_action"] == "L12"
    assert model.intermediate_layers[-2:] == [8, 12]


def test_streampetr_teacher_forcing_uses_previous_gt_then_predictions():
    model = make_fake_petr3d(temporal_memory=dict(enabled=True))
    model.temporal_memory["policy_offline_features"] = "teacher_forcing"
    state = model.temporal_memory_controller.states.setdefault(
        ("scene-a", 0), {})
    state["offline_gt_features"] = {"num_annotations": 7.0}
    model._temporal_memory_record_predictions(
        [{"boxes_3d": torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0, 4.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0],
            [2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ])}],
        [{"scene_token": "scene-a"}],
    )

    model.train(False)
    training = model._temporal_memory_policy_features(
        {"scene_token": "scene-a"}, {}, state, frame_number=1,
        training=True)
    inference = model._temporal_memory_policy_features(
        {"scene_token": "scene-a"}, {}, state, frame_number=1,
        training=False)

    assert training["num_annotations"] == 7.0
    assert inference["num_annotations"] == 3.0
    assert inference["max_object_speed_mps"] == 5.0


def test_streampetr_prediction_state_is_causal_and_tracks_transitions():
    model = make_fake_petr3d(temporal_memory=dict(enabled=True))
    model.temporal_memory["policy_offline_features"] = "teacher_forcing"
    state = model.temporal_memory_controller.states.setdefault(
        ("scene-a", 0), {})
    first = dict(
        boxes_3d=torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0],
        ]),
        scores_3d=torch.tensor([0.9, 0.8]),
        labels_3d=torch.tensor([1, 2]),
    )
    model._temporal_memory_record_predictions(
        [first], [{"scene_token": "scene-a"}])
    before_second_decode = model._temporal_memory_policy_features(
        {"scene_token": "scene-a"}, {}, state, frame_number=1,
        training=False)
    second = dict(
        boxes_3d=torch.tensor([
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0, 0.0],
            [30.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 4.0, 0.0],
        ]),
        scores_3d=torch.tensor([0.95, 0.7]),
        labels_3d=torch.tensor([1, 3]),
    )
    model._temporal_memory_record_predictions(
        [second], [{"scene_token": "scene-a"}])
    after_second_decode = model._temporal_memory_policy_features(
        {"scene_token": "scene-a"}, {}, state, frame_number=2,
        training=False)

    assert before_second_decode["num_annotations"] == 2.0
    assert before_second_decode["new_object_proxy_count"] == 2.0
    assert after_second_decode["num_annotations"] == 2.0
    assert after_second_decode["matched_object_proxy_count"] == 1.0
    assert after_second_decode["new_object_proxy_count"] == 1.0
    assert after_second_decode["disappeared_object_proxy_count"] == 1.0
    assert state["prediction_labels"].tolist() == [1, 3]
    assert torch.equal(state["prediction_scores"], torch.tensor([0.95, 0.7]))


def test_streampetr_transition_matching_uses_global_centers_and_same_class():
    model = make_fake_petr3d(temporal_memory=dict(enabled=True))
    state = model.temporal_memory_controller.states.setdefault(
        ("scene-a", 0), {})
    identity = torch.eye(4)
    translated = torch.eye(4)
    translated[0, 3] = 10.0
    first = dict(
        boxes_3d=torch.tensor([
            [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]),
        scores_3d=torch.tensor([0.9, 0.9]),
        labels_3d=torch.tensor([1, 2]),
    )
    second = dict(
        boxes_3d=torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]),
        scores_3d=torch.tensor([0.9, 0.9]),
        labels_3d=torch.tensor([1, 3]),
    )
    model._temporal_memory_record_predictions(
        [first], [{"scene_token": "scene-a", "ego_pose": identity}])
    model._temporal_memory_record_predictions(
        [second], [{"scene_token": "scene-a", "ego_pose": translated}])

    assert state["prediction_features"]["matched_object_proxy_count"] == 1.0
    assert state["prediction_features"]["new_object_proxy_count"] == 1.0
    assert state["prediction_features"]["disappeared_object_proxy_count"] == 1.0
    assert state["prediction_boxes"][:, 0].tolist() == [10.0, 0.0]


def test_streampetr_gt_transitions_use_global_centers_and_same_class():
    model = make_fake_petr3d(temporal_memory=dict(enabled=True))
    identity = torch.eye(4)
    translated = torch.eye(4)
    translated[0, 3] = 10.0
    previous_local = torch.tensor([
        [10.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ])
    previous_global = model._temporal_memory_global_boxes(
        previous_local, identity)
    current_local = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [-10.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ])

    features = model._temporal_memory_offline_features(
        {}, dict(
            gt_bboxes_3d=current_local,
            gt_labels_3d=torch.tensor([1, 3]),
            ego_pose=translated),
        previous_boxes=previous_global,
        previous_labels=torch.tensor([1, 2]))

    assert features["matched_object_proxy_count"] == 1.0
    assert features["new_object_proxy_count"] == 1.0
    assert features["disappeared_object_proxy_count"] == 1.0


def test_streampetr_policy_supervision_returns_trainable_loss():
    from projects.mmdet3d_plugin.models.temporal_memory import (
        MlpFramePolicy,
        TemporalMemoryController,
    )

    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True,
        updater="reuse",
        keyframe_interval=4,
    ))
    model.temporal_memory["policy_supervision"] = dict(
        enabled=True,
        target="similarity",
        thresholds=(0.9, 0.7, 0.5),
        weight=1.0,
    )
    policy = MlpFramePolicy(
        checkpoint=None,
        load_weights=False,
        feature_names=["ego_speed_mps"],
        labels=["easy", "medium", "hard", "very_hard"],
        actions=dict(
            easy="L8", medium="L12", hard="L16", very_hard="L24"),
    )
    model.temporal_memory_controller = TemporalMemoryController(
        enabled=True, updater="reuse", keyframe_interval=4, policy=policy)
    model.temporal_memory_policy = policy
    model.train(True)
    model._extract_img_feat_temporal_memory(
        _frame(1.0),
        [{"scene_token": "scene-a", "ego_speed_mps": 1.0}],
        training=True,
    )

    _, losses = model._extract_img_feat_temporal_memory(
        _frame(2.0),
        [{"scene_token": "scene-a", "ego_speed_mps": 2.0}],
        training=True,
    )

    assert losses["loss_tm_policy"].requires_grad
    losses["loss_tm_policy"].backward()
    assert any(param.grad is not None for param in policy.parameters())


def test_streampetr_online_policy_features_cover_configured_motion_schema():
    model = make_fake_petr3d(temporal_memory=dict(enabled=True))
    previous = dict(
        memory=torch.ones(1),
        policy_state=dict(
            timestamp=1.0,
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_yaw_rad=0.0,
            ego_speed_mps=0.0,
            yaw_rate_deg_s=0.0,
        ),
    )
    pose = torch.eye(4)
    pose[0, 0] = 0.0
    pose[0, 1] = -1.0
    pose[1, 0] = 1.0
    pose[1, 1] = 0.0
    pose[0, 3] = 3.0
    pose[1, 3] = 4.0
    pose[2, 3] = 5.0

    features = model._temporal_memory_policy_features(
        {"scene_token": "scene-a", "frame_idx": 8},
        {"timestamp": 3.0, "ego_pose": pose},
        previous,
        frame_number=2,
        training=False,
    )

    expected = {
        "frame_number", "info_frame_idx", "is_scene_start", "dt_s",
        "ego_x_m", "ego_y_m", "ego_z_m", "ego_yaw_deg",
        "delta_ego_translation_m", "ego_speed_mps", "ego_accel_mps2",
        "yaw_rate_deg_s", "abs_yaw_rate_deg_s", "yaw_accel_deg_s2",
        "abs_yaw_accel_deg_s2",
    }
    assert expected <= features.keys()
    assert features["dt_s"] == 2.0
    assert features["delta_ego_translation_m"] == 5.0
    assert features["ego_speed_mps"] == 2.5
    assert features["ego_accel_mps2"] == 1.25
    assert features["ego_yaw_deg"] == pytest.approx(90.0)
    assert features["yaw_rate_deg_s"] == pytest.approx(45.0)
    assert features["yaw_accel_deg_s2"] == pytest.approx(22.5)


def test_streampetr_offline_features_cover_declared_gt_schema():
    model = make_fake_petr3d(temporal_memory=dict(enabled=True))
    boxes = torch.tensor([
        [3.0, 4.0, 0.0, 2.0, 3.0, 4.0, 0.0, 3.0, 4.0],
        [6.0, 8.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 2.0],
    ])
    camera_boxes = [
        torch.tensor([[0.0, 0.0, 10.0, 20.0]]),
        torch.tensor([[0.0, 0.0, 20.0, 30.0], [0.0, 0.0, 5.0, 10.0]]),
    ]

    features = model._temporal_memory_offline_features(
        {}, {"gt_bboxes_3d": boxes, "gt_bboxes": camera_boxes})

    declared = {
        "num_annotations", "avg_object_distance_m",
        "median_object_distance_m", "min_object_distance_m",
        "p10_object_distance_m", "p90_object_distance_m",
        "near_object_count_10m", "near_object_count_20m",
        "near_object_count_30m", "near_object_count_50m",
        "avg_box_length_m", "avg_box_width_m", "avg_box_height_m",
        "avg_box_volume_m3", "avg_box_area_m2",
        "avg_object_speed_mps", "max_object_speed_mps",
        "camera_annotation_count", "max_camera_annotation_count",
        "min_camera_annotation_count", "avg_camera_annotation_count",
        "avg_2d_box_area_px", "max_2d_box_area_px",
        "new_object_proxy_count", "matched_object_proxy_count",
        "disappeared_object_proxy_count", "new_object_proxy_fraction",
    }
    assert declared <= features.keys()
    assert features["num_annotations"] == 2.0
    assert features["avg_object_distance_m"] == 7.5
    assert features["max_object_speed_mps"] == 5.0
    assert features["camera_annotation_count"] == 3.0
    assert features["max_2d_box_area_px"] == 600.0


def test_streampetr_prediction_features_threshold_scores_before_statistics():
    model = make_fake_petr3d(temporal_memory=dict(enabled=True))
    model.temporal_memory["policy_prediction_score_thr"] = 0.5
    boxes = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 0.0, 30.0, 40.0],
    ])

    features = model._temporal_memory_prediction_features(dict(
        boxes_3d=boxes,
        scores_3d=torch.tensor([0.9, 0.1]),
    ))

    assert features["num_annotations"] == 1.0
    assert features["max_object_speed_mps"] == 5.0
    assert features["avg_box_length_m"] == 1.0


def test_streampetr_parameter_anchor_connects_every_trainable_tm_module():
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D

    model = Petr3D.__new__(Petr3D)
    torch.nn.Module.__init__(model)
    model.temporal_memory_projector = torch.nn.Linear(2, 2)
    model.temporal_memory_updater = torch.nn.Linear(2, 2)
    model.temporal_memory_policy = torch.nn.Linear(2, 2)

    loss = model._temporal_memory_parameter_anchor_loss()
    loss.backward()

    assert loss.item() == 0.0
    for module in (
        model.temporal_memory_projector,
        model.temporal_memory_updater,
        model.temporal_memory_policy,
    ):
        assert all(param.grad is not None for param in module.parameters())


def test_streampetr_policy_supervision_schedule_bounds_teacher_calls():
    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True, updater="reuse", keyframe_interval=99))
    model.temporal_memory["policy_supervision"] = dict(
        enabled=True, target="similarity", warmup_ratio=0.1,
        max_train_steps=100)
    state = model.temporal_memory_controller.states.setdefault(
        ("scene-a", 0), dict(
            scene_token="scene-a", frame_index=1, frames_since_full=0,
            memory=_frame(101.0)))
    model.train(True)
    model._tm_train_step = 10
    model._extract_img_feat_temporal_memory(
        _frame(2.0), _metas(), training=True)
    assert model.full_calls == 0
    model._tm_train_step = 9
    model._extract_img_feat_temporal_memory(
        _frame(3.0), _metas(), training=True)
    assert model.full_calls == 1
    model.train(False)
    model._extract_img_feat_temporal_memory(
        _frame(4.0), _metas(), training=False)
    assert model.full_calls == 1


def test_streampetr_policy_supervision_step_survives_state_dict_resume():
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D

    first = Petr3D.__new__(Petr3D)
    torch.nn.Module.__init__(first)
    first.temporal_memory = dict(policy_supervision=dict(
        enabled=True, warmup_ratio=0.1, max_train_steps=100))
    first.register_buffer("_tm_train_step_buffer", torch.tensor(9, dtype=torch.long))
    payload = first.state_dict()

    resumed = Petr3D.__new__(Petr3D)
    torch.nn.Module.__init__(resumed)
    resumed.temporal_memory = first.temporal_memory
    resumed.register_buffer("_tm_train_step_buffer", torch.tensor(0, dtype=torch.long))
    resumed.load_state_dict(payload)

    assert int(resumed._tm_train_step_buffer) == 9
    assert resumed._temporal_memory_policy_supervision_active(training=True)
    resumed._tm_train_step_buffer.add_(1)
    assert not resumed._temporal_memory_policy_supervision_active(training=True)


def test_streampetr_policy_class_weights_change_single_sample_and_queue_loss():
    model = make_fake_petr3d(temporal_memory=dict(enabled=True))
    losses = [torch.tensor(2.0), torch.tensor(4.0)]

    single = model._temporal_memory_aggregate_policy_losses(
        [(losses[0], 3.0)])
    weighted = model._temporal_memory_aggregate_policy_losses(
        [(losses[0], 1.0), (losses[1], 3.0)])

    assert single.item() == pytest.approx(6.0)
    assert weighted.item() == pytest.approx(7.0)


def test_streampetr_adaptive_motion_reset_forces_full_per_sample():
    model = make_fake_petr3d(temporal_memory=dict(
        enabled=True,
        keyframe_interval=99,
        updater="reuse",
    ))
    model.temporal_memory_controller.adaptive_reset = dict(
        enabled=True, max_frames_between_full=2, ego_speed_mps=2.0)
    model.temporal_memory_controller.max_frames_between_full = 4
    model._extract_img_feat_temporal_memory(
        _frame(1.0), _metas(), training=False)
    model._extract_img_feat_temporal_memory(
        _frame(2.0),
        [{"scene_token": "scene-a", "ego_speed_mps": 0.5}],
        training=False,
    )

    features, _ = model._extract_img_feat_temporal_memory(
        _frame(3.0),
        [{"scene_token": "scene-a", "ego_speed_mps": 3.0}],
        training=False,
    )

    assert torch.equal(features, _frame(103.0))
    assert model.full_calls == 2


def test_streampetr_freeze_keeps_only_temporal_modules_trainable():
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D

    model = Petr3D.__new__(Petr3D)
    torch.nn.Module.__init__(model)
    model.img_backbone = torch.nn.Linear(2, 2)
    model.pts_bbox_head = torch.nn.Linear(2, 2)
    model.temporal_memory_projector = torch.nn.Linear(2, 2)
    model.temporal_memory_updater = torch.nn.Linear(2, 2)
    model.temporal_memory_policy = torch.nn.Linear(2, 2)

    model._freeze_temporal_memory_pretrained_detector()

    assert not any(param.requires_grad for param in model.img_backbone.parameters())
    assert not any(param.requires_grad for param in model.pts_bbox_head.parameters())
    assert all(param.requires_grad for param in model.temporal_memory_projector.parameters())
    assert all(param.requires_grad for param in model.temporal_memory_updater.parameters())
    assert all(param.requires_grad for param in model.temporal_memory_policy.parameters())


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

    def route(self, img, img_metas, training=False, prev_exists=None,
              policy_data=None):
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

    def route(self, img, img_metas, training=False, prev_exists=None,
              policy_data=None):
        seen["route"] = (img, img_metas, training, prev_exists)
        return img + 9.0, {}

    def simple_test_pts(self, img_metas, **data):
        seen["head"] = (img_metas, data["img_feats"])
        return ["boxes"]

    model._extract_img_feat_temporal_memory = MethodType(route, model)
    model.simple_test_pts = MethodType(simple_test_pts, model)
    model.temporal_memory_controller = None
    img = _frame(4.0)
    metas = _metas()

    results = model.simple_test(metas, img=img)

    assert seen["route"] == (img, metas, False, None)
    assert seen["head"][0] is metas
    assert torch.equal(seen["head"][1], img + 9.0)
    assert results == [{"pts_bbox": "boxes"}]
