import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_repdetr3d_non_reuse_temporal_path_projects_partial_features():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D

    class Controller:
        def will_use_keyframe(self, scene_token=None, prev_exists=None):
            return False

        def get_or_update(self, extract_full, **kwargs):
            self.kwargs = kwargs
            self.full = extract_full
            return "updated"

    class Projector:
        def __call__(self, partial):
            self.partial = partial
            return "projected"

    model = RepDetr3D.__new__(RepDetr3D)
    model.training = False
    model.temporal_memory_enabled = True
    model.temporal_memory = {
        "updater": "partial_only",
        "partial_layer": "block8",
    }
    model.tm_profiler = None
    model.temporal_memory_projector = Projector()
    model.temporal_memory_controller = Controller()
    model._profile_extract_partial_features = lambda data, layer_id: ("partial", layer_id)
    model._profile_extract_full_features = lambda data, len_queue, training_mode: "full"

    out = model._extract_img_feat_temporal_memory(
        [{"scene_token": "scene-a"}],
        img=torch.randn(1, 6, 3, 32, 32),
        prev_exists=torch.ones(1),
    )

    assert out == "updated"
    assert model.temporal_memory_projector.partial == ("partial", "block8")
    assert model.temporal_memory_controller.kwargs["partial_current"] == "projected"
    assert model.temporal_memory_controller.kwargs["scene_token"] == "scene-a"
    assert model.temporal_memory_controller.kwargs["prev_exists"] is True


def test_repdetr3d_depth_router_metadata_is_per_sample_and_rejects_gt_keys():
    import pytest
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D

    model = RepDetr3D.__new__(RepDetr3D)
    model.temporal_memory = dict(router=dict(feature_names=("speed", "age")))

    metadata = model._tm_depth_router_metadata_tensor([
        {"speed": 1.0, "age": 2.0},
        {"speed": 3.0, "age": 4.0},
    ], device=torch.device("cpu"), dtype=torch.float32)

    assert metadata.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="online-only"):
        model._tm_depth_router_metadata_tensor(
            [{"speed": 1.0, "gt_bboxes_3d": 1.0}],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_repdetr3d_depth_router_groups_continuations_and_restores_sample_order():
    """Catches a branch that treats per-sample router actions as per-batch."""
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D

    class Backbone:
        def __init__(self):
            self.calls = []

        def forward_blocks(self, tokens, start_index, stop_at):
            self.calls.append((tokens[:, 0, 0, 0].tolist(), start_index, stop_at))
            return tokens + float(stop_at + 1)

    model = RepDetr3D.__new__(RepDetr3D)
    model.img_backbone = Backbone()
    model._project_temporal_memory_partial = lambda features, **kwargs: [features]

    # B=4, N_cam=2.  The first coordinate carries the original sample index.
    l8_tokens = torch.zeros(8, 1, 1, 1)
    l8_tokens[:, 0, 0, 0] = torch.arange(8)
    l8_projected = [torch.arange(8, dtype=torch.float32).view(4, 2, 1, 1, 1)]

    partials = model._tm_depth_router_grouped_partial_features(
        l8_tokens=l8_tokens,
        batch_size=4,
        num_cams=2,
        actions=torch.tensor([0, 1, 3, 2]),  # L8, L12, L24, L16
        l8_projected=l8_projected,
    )

    assert partials[0][0][:, 0, 0, 0, 0].tolist() == [0.0]
    assert partials[1][0][:, 0, 0, 0, 0].tolist() == [14.0]
    assert partials[2][0][:, 0, 0, 0, 0].tolist() == [22.0]
    assert 3 not in partials
    assert model.img_backbone.calls == [([2.0, 3.0], 8, 11), ([6.0, 7.0], 8, 15)]


def test_repdetr3d_depth_router_executes_one_action_per_sample_and_returns_batch_order():
    """Catches a router path that accidentally applies one route to all samples."""
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D
    from projects.mmdet3d_plugin.models.temporal_memory.routers import DepthRouteOutput

    class Router:
        def __call__(self, **kwargs):
            logits = torch.zeros(4, 4, requires_grad=True)
            probabilities = logits.softmax(dim=-1)
            hard = torch.eye(4, dtype=torch.float32)[torch.tensor([0, 1, 3, 2])]
            return DepthRouteOutput(
                logits=logits,
                probabilities=probabilities,
                hard_actions=torch.tensor([0, 1, 3, 2]),
                hard_one_hot=hard,
                expected_cost=probabilities.new_tensor(0.625),
            )

    class Controller:
        def __init__(self):
            self.states = {
                (scene, index): dict(
                    scene_token=scene,
                    frames_since_full=0,
                    memory=[torch.ones(1, 2, 1, 1, 1)],
                )
                for index, scene in enumerate(('a', 'b', 'c', 'd'))
            }

        def get_or_update_samples(self, requests, training=False):
            self.requests = requests
            return [
                request['extract_full']() if request['action'] == 'L24'
                else request['partial_current']
                for request in requests
            ]

    model = RepDetr3D.__new__(RepDetr3D)
    model.temporal_memory = dict(router=dict(feature_names=('speed',), target_cost=0.5))
    model.temporal_memory_depth_router = Router()
    model.temporal_memory_controller = Controller()
    model._tm_depth_router_prepare_l8 = lambda data: (
        torch.zeros(8, 1, 1, 1), [torch.ones(4, 2, 1, 1, 1)])
    model._tm_depth_router_grouped_partial_features = lambda **kwargs: {
        0: [torch.full((1, 2, 1, 1, 1), 8.0)],
        1: [torch.full((1, 2, 1, 1, 1), 12.0)],
        2: [torch.full((1, 2, 1, 1, 1), 16.0)],
    }
    model._tm_depth_router_extract_full_group = lambda data, indices, training: [
        torch.full((len(indices), 2, 1, 1, 1), 24.0)]

    features, losses = model._tm_route_depth_batch(
        dict(img=torch.randn(4, 2, 3, 8, 8)),
        [
            {'scene_token': 'a', 'speed': 1.0},
            {'scene_token': 'b', 'speed': 2.0},
            {'scene_token': 'c', 'speed': 3.0},
            {'scene_token': 'd', 'speed': 4.0},
        ],
        training=True,
    )

    assert [request['action'] for request in model.temporal_memory_controller.requests] == [
        'L8', 'L12', 'L24', 'L16']
    assert features[0][:, 0, 0, 0, 0].tolist() == [8.0, 12.0, 24.0, 16.0]
    assert set(losses) == {'loss_tm_budget', 'loss_tm_balance'}


def test_repdetr3d_training_temporal_path_stacks_features_and_distill_loss():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    full0 = [torch.zeros(1, 2, 4, 3, 5)]
    full1 = [torch.ones(1, 2, 4, 3, 5)]
    partial1 = [torch.full((1, 2, 4, 3, 5), 0.25, requires_grad=True)]

    class Projector:
        def __call__(self, partial):
            return partial

    model = RepDetr3D.__new__(RepDetr3D)
    model.temporal_memory_enabled = True
    model.temporal_memory = {
        "updater": "partial_only",
        "partial_layer": "block8",
        "train_with_teacher": True,
        "use_distillation": True,
    }
    model.tm_profiler = None
    model.temporal_memory_projector = Projector()
    model.temporal_memory_controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=2,
        updater="partial_only",
        train_with_teacher=True,
        distill_weight=1.0,
    )
    model._profile_extract_full_features = lambda frame_data, len_queue, training_mode: (
        full0 if frame_data["img"].sum().item() == 0 else full1
    )
    model._profile_extract_partial_features = lambda frame_data, layer_id: partial1

    img = torch.stack([
        torch.zeros(1, 2, 3, 4, 4),
        torch.ones(1, 2, 3, 4, 4),
    ], dim=1)
    features, losses = model._extract_img_feat_temporal_memory_train(
        [
            [{"scene_token": "scene-a"}],
            [{"scene_token": "scene-a"}],
        ],
        img=img,
        prev_exists=torch.ones(1, 2),
    )

    assert [tuple(feature.shape) for feature in features] == [(1, 2, 2, 4, 3, 5)]
    assert torch.equal(features[0][:, 0], full0[0])
    assert torch.equal(features[0][:, 1], partial1[0])
    assert "loss_tm_distill" in losses
    losses["loss_tm_distill"].backward()
    assert partial1[0].grad is not None


def test_repdetr3d_training_temporal_path_keeps_distill_loss_key_without_non_keyframes():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    full0 = [torch.zeros(1, 2, 4, 3, 5)]

    model = RepDetr3D.__new__(RepDetr3D)
    model.temporal_memory_enabled = True
    model.temporal_memory = {
        "updater": "partial_only",
        "partial_layer": "block8",
        "train_with_teacher": True,
        "use_distillation": True,
    }
    model.tm_profiler = None
    model.temporal_memory_controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=1,
        updater="partial_only",
        train_with_teacher=True,
        distill_weight=1.0,
    )
    model._profile_extract_full_features = lambda frame_data, len_queue, training_mode: full0

    _, losses = model._extract_img_feat_temporal_memory_train(
        [[{"scene_token": "scene-a"}]],
        img=torch.zeros(1, 1, 2, 3, 4, 4),
        prev_exists=torch.ones(1, 1),
    )

    assert "loss_tm_distill" in losses
    assert losses["loss_tm_distill"].item() == 0


def test_repdetr3d_temporal_memory_anchor_loss_tracks_trainable_tm_params():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D

    model = RepDetr3D.__new__(RepDetr3D)
    torch.nn.Module.__init__(model)
    model.temporal_memory_projector = torch.nn.Linear(2, 2)
    model.temporal_memory_updater = torch.nn.Linear(2, 2)
    model.temporal_memory_policy = torch.nn.Linear(2, 2)

    loss = model._temporal_memory_parameter_anchor_loss()

    assert loss.item() == 0
    loss.backward()
    for param in list(model.temporal_memory_projector.parameters()) + list(
        model.temporal_memory_updater.parameters()
    ) + list(
        model.temporal_memory_policy.parameters()
    ):
        assert param.grad is not None
        assert torch.equal(param.grad, torch.zeros_like(param.grad))


def test_repdetr3d_temporal_memory_freeze_keeps_only_tm_modules_trainable():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D

    model = RepDetr3D.__new__(RepDetr3D)
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


def test_repdetr3d_mlp_policy_detects_previous_l12_similarity_feature():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D
    from projects.mmdet3d_plugin.models.temporal_memory.policies import build_frame_policy

    model = RepDetr3D.__new__(RepDetr3D)
    model.temporal_memory_controller = type(
        "Controller",
        (),
        {"policy": build_frame_policy(dict(
            type="mlp",
            checkpoint=None,
            load_weights=False,
            feature_names=["previous_l12_similarity"],
            labels=["easy", "medium", "hard", "very_hard"],
            actions={"easy": "L8", "medium": "L12", "hard": "L16", "very_hard": "L24"},
        ))},
    )()
    model.temporal_memory = {}

    assert model._temporal_memory_policy_needs_previous_l12_drift()


def test_repdetr3d_commit_policy_state_tracks_previous_l12_similarity():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D

    model = RepDetr3D.__new__(RepDetr3D)
    first = [torch.ones(1, 1, 2, 1, 1)]
    second = [-torch.ones(1, 1, 2, 1, 1)]
    state0 = {"scene_token": "scene-a", "is_scene_start": True}
    state1 = {"scene_token": "scene-a", "is_scene_start": False}

    model._temporal_memory_commit_policy_state({"_state": state0}, l12_features=first)
    model._temporal_memory_commit_policy_state({"_state": state1}, l12_features=second)

    assert torch.isclose(torch.tensor(model._tm_policy_previous_l12_similarity), torch.tensor(-1.0))


def test_repdetr3d_policy_supervision_labels_similarity_into_four_classes():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D

    model = RepDetr3D.__new__(RepDetr3D)
    model.temporal_memory = dict(
        policy_supervision=dict(thresholds=(0.85, 0.75, 0.65)))

    assert model._temporal_memory_policy_target_index(0.90) == 0
    assert model._temporal_memory_policy_target_index(0.80) == 1
    assert model._temporal_memory_policy_target_index(0.70) == 2
    assert model._temporal_memory_policy_target_index(0.60) == 3


def test_repdetr3d_policy_supervision_loss_uses_similarity_target():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D
    from projects.mmdet3d_plugin.models.temporal_memory.policies import build_frame_policy

    model = RepDetr3D.__new__(RepDetr3D)
    torch.nn.Module.__init__(model)
    model.temporal_memory = dict(
        policy_supervision=dict(thresholds=(0.85, 0.75, 0.65), weight=2.0))
    model.temporal_memory_policy = build_frame_policy(dict(
        type="mlp",
        checkpoint=None,
        load_weights=False,
        feature_names=["speed"],
        hidden_dim=4,
        labels=["easy", "medium", "hard", "very_hard"],
        actions={"easy": "L8", "medium": "L12", "hard": "L16", "very_hard": "L24"},
    ))
    previous = [torch.ones(1, 1, 2, 1, 1)]
    current = [torch.ones(1, 1, 2, 1, 1)]

    loss = model._temporal_memory_policy_supervision_loss(
        {"speed": 1.0}, previous, current)

    assert loss.shape == ()
    assert loss.requires_grad


def test_repdetr3d_policy_supervision_loss_uses_class_weights():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D
    from projects.mmdet3d_plugin.models.temporal_memory.policies import build_frame_policy

    model = RepDetr3D.__new__(RepDetr3D)
    torch.nn.Module.__init__(model)
    model.temporal_memory = dict(
        policy_supervision=dict(
            thresholds=(0.9, 0.76, 0.675),
            weight=2.0,
            class_weights=(1.0, 1.0, 1.0, 3.0),
        ))
    model.temporal_memory_policy = build_frame_policy(dict(
        type="mlp",
        checkpoint=None,
        load_weights=False,
        feature_names=["speed"],
        hidden_dim=4,
        labels=["easy", "medium", "hard", "very_hard"],
        actions={"easy": "L8", "medium": "L12", "hard": "L16", "very_hard": "L24"},
    ))
    previous = [torch.ones(1, 1, 2, 1, 1)]
    current = [-torch.ones(1, 1, 2, 1, 1)]
    logits = model.temporal_memory_policy.logits({"speed": 1.0})
    target = torch.as_tensor([3], dtype=torch.long, device=logits.device)
    expected = torch.nn.functional.cross_entropy(logits, target) * 3.0 * 2.0

    loss = model._temporal_memory_policy_supervision_loss(
        {"speed": 1.0}, previous, current)

    assert torch.allclose(loss, expected)


def test_repdetr3d_policy_supervision_zero_loss_keeps_ddp_loss_key_when_no_pairs():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D

    model = RepDetr3D.__new__(RepDetr3D)
    torch.nn.Module.__init__(model)
    model.temporal_memory = dict(policy_supervision=dict(enabled=True))

    losses = model._temporal_memory_aux_losses(
        features=[torch.ones(1, 1, 2, 1, 1)],
        distill_losses=[],
        policy_losses=[],
    )

    assert set(losses) == {"loss_tm_policy"}
    assert torch.equal(losses["loss_tm_policy"], torch.zeros(()))


def test_repdetr3d_policy_supervision_can_score_identical_features_with_cka():
    from projects.mmdet3d_plugin.models.detectors.repdetr3d import RepDetr3D

    model = RepDetr3D.__new__(RepDetr3D)
    model.temporal_memory = dict(policy_supervision=dict(target="cka"))
    features = [torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 2, 2)]

    score = model._temporal_memory_policy_target_score(features, features)

    assert torch.isclose(score, torch.tensor(1.0), atol=1e-5)
