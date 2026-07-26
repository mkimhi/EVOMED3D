import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_reuse_controller_k1_always_runs_full_path():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=1,
        updater="reuse",
    )

    calls = []

    def full():
        calls.append(len(calls))
        return f"full-{len(calls)}"

    outputs = [
        controller.get_or_update(full, scene_token="scene-a"),
        controller.get_or_update(full, scene_token="scene-a"),
        controller.get_or_update(full, scene_token="scene-a"),
    ]

    assert outputs == ["full-1", "full-2", "full-3"]
    assert calls == [0, 1, 2]


def test_reuse_controller_reuses_between_fixed_keyframes():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=3,
        updater="reuse",
    )

    calls = []

    def full():
        calls.append(len(calls))
        return f"full-{len(calls)}"

    outputs = [
        controller.get_or_update(full, scene_token="scene-a"),
        controller.get_or_update(full, scene_token="scene-a"),
        controller.get_or_update(full, scene_token="scene-a"),
        controller.get_or_update(full, scene_token="scene-a"),
    ]

    assert outputs == ["full-1", "full-1", "full-1", "full-2"]
    assert calls == [0, 1]


def test_reuse_controller_resets_on_scene_boundary_and_discontinuity():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=5,
        updater="reuse",
    )

    calls = []

    def full():
        calls.append(len(calls))
        return f"full-{len(calls)}"

    assert controller.get_or_update(full, scene_token="a") == "full-1"
    assert controller.get_or_update(full, scene_token="a") == "full-1"
    assert controller.get_or_update(full, scene_token="b") == "full-2"
    assert controller.get_or_update(full, scene_token="b", prev_exists=False) == "full-3"
    assert calls == [0, 1, 2]


def test_reuse_controller_debug_logs_keyframe_decisions(capsys):
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=2,
        updater="reuse",
        debug=True,
    )

    def full():
        return "full"

    controller.get_or_update(full, scene_token="scene-a")
    controller.get_or_update(full, scene_token="scene-a")

    out = capsys.readouterr().out
    assert "decision=keyframe" in out
    assert "decision=non_keyframe updater=reuse" in out


def test_controller_forwards_partial_features_and_ego_embedding_to_updater():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    class Updater:
        def __call__(self, previous_memory, partial_current=None, ego_motion_embedding=None):
            self.args = (previous_memory, partial_current, ego_motion_embedding)
            return "updated"

    updater = Updater()
    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=3,
        updater=updater,
    )

    def full():
        return "full"

    assert controller.get_or_update(full, scene_token="scene-a") == "full"
    assert controller.get_or_update(
        full,
        scene_token="scene-a",
        partial_current="partial",
        ego_motion_embedding="ego",
    ) == "updated"
    assert updater.args == ("full", "partial", "ego")


def test_controller_reports_next_frame_keyframe_without_mutating_state():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=3,
        updater="reuse",
    )

    assert controller.will_use_keyframe(scene_token="scene-a") is True
    assert controller.frame_index == 0
    controller.get_or_update(lambda: "full", scene_token="scene-a")
    assert controller.will_use_keyframe(scene_token="scene-a") is False
    assert controller.will_use_keyframe(scene_token="scene-b") is True
    assert controller.will_use_keyframe(scene_token="scene-a", prev_exists=False) is True


def test_controller_accepts_batched_prev_exists_tensor():
    import torch
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=3,
        updater="reuse",
    )

    assert controller.get_or_update(lambda: "full-1", scene_token="scene-a") == "full-1"
    assert controller.get_or_update(
        lambda: "full-2",
        scene_token="scene-a",
        prev_exists=torch.tensor([True, True]),
    ) == "full-1"
    assert controller.will_use_keyframe(
        scene_token="scene-a",
        prev_exists=torch.tensor([True, False]),
    ) is True
    assert controller.get_or_update(
        lambda: "full-3",
        scene_token="scene-a",
        prev_exists=torch.tensor([True, False]),
    ) == "full-3"


def test_fixed_frame_policy_emits_l24_every_k_frames_and_partial_between():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=3,
        updater="reuse",
        policy=dict(type="fixed_interval", partial_action="L8"),
    )

    assert controller.peek_action(scene_token="scene-a") == "L24"
    controller.get_or_update(lambda: "full-0", scene_token="scene-a")
    assert controller.peek_action(scene_token="scene-a") == "L8"
    controller.get_or_update(lambda: "unused", scene_token="scene-a")
    assert controller.peek_action(scene_token="scene-a") == "L8"
    controller.get_or_update(lambda: "unused", scene_token="scene-a")
    assert controller.peek_action(scene_token="scene-a") == "L24"


def test_controller_passes_policy_action_to_layered_updater():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    class Updater:
        def __call__(self, previous_memory, partial_current=None, action=None, **kwargs):
            self.action = action
            return partial_current

    updater = Updater()
    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=2,
        updater=updater,
        policy=dict(type="fixed_interval", partial_action="L12"),
    )

    assert controller.get_or_update(lambda: "full", scene_token="scene-a") == "full"
    assert controller.get_or_update(
        lambda: "unused",
        scene_token="scene-a",
        partial_current="partial",
    ) == "partial"
    assert updater.action == "L12"


def test_cycle_frame_policy_emits_configured_action_sequence():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=4,
        updater="reuse",
        policy=dict(type="cycle", actions=("L24", "L8", "L12", "L16")),
    )

    actions = []
    for _ in range(6):
        actions.append(controller.peek_action(scene_token="scene-a"))
        controller.get_or_update(lambda: "full", scene_token="scene-a")

    assert actions == ["L24", "L8", "L12", "L16", "L24", "L8"]


def test_cycle_frame_policy_forces_l24_on_scene_reset_then_restarts_cycle():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=4,
        updater="reuse",
        policy=dict(type="cycle", actions=("L24", "L8", "L12", "L16")),
    )

    controller.get_or_update(lambda: "full-a", scene_token="scene-a")
    controller.get_or_update(lambda: "unused", scene_token="scene-a")

    assert controller.peek_action(scene_token="scene-b") == "L24"
    controller.get_or_update(lambda: "full-b", scene_token="scene-b")
    assert controller.peek_action(scene_token="scene-b") == "L8"



def test_mlp_frame_policy_maps_checkpoint_classes_to_actions(tmp_path):
    import torch
    from projects.mmdet3d_plugin.models.temporal_memory.policies import build_frame_policy

    checkpoint = tmp_path / "policy.pt"
    torch.save({
        "feature_names": ["speed"],
        "mean": [0.0],
        "scale": [1.0],
        "hidden_dim": 2,
        "labels": ["easy", "medium", "hard"],
        "actions": {"easy": "L8", "medium": "L12", "hard": "L24"},
        "state_dict": {
            "0.weight": torch.tensor([[1.0], [-1.0]]),
            "0.bias": torch.tensor([0.0, 0.0]),
            "2.weight": torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
            "2.bias": torch.tensor([0.0, 0.0, 0.0]),
        },
    }, checkpoint)

    policy = build_frame_policy(dict(type="mlp", checkpoint=str(checkpoint)))

    assert policy.decide(1, True, policy_features={"speed": 2.0}) == "L8"
    assert policy.decide(1, True, policy_features={"speed": -2.0}) == "L24"
    assert policy.decide(0, False, policy_features={"speed": 2.0}) == "L24"


def test_mlp_frame_policy_can_skip_checkpoint_loading():
    from projects.mmdet3d_plugin.models.temporal_memory.policies import build_frame_policy

    policy = build_frame_policy(dict(
        type="mlp",
        checkpoint=None,
        load_weights=False,
        feature_names=["speed"],
        hidden_dim=4,
        labels=["easy", "medium", "hard"],
        actions={"easy": "L8", "medium": "L12", "hard": "L24"},
    ))

    assert policy.decide(0, False, policy_features={"speed": 1.0}) == "L24"
    assert policy.decide(1, True, policy_features={"speed": 1.0}) in {"L8", "L12", "L24"}


def test_mlp_frame_policy_uses_explicit_normalization_without_checkpoint():
    import torch
    from projects.mmdet3d_plugin.models.temporal_memory.policies import build_frame_policy

    policy = build_frame_policy(dict(
        type="mlp",
        checkpoint=None,
        load_weights=False,
        feature_names=["frame_number"],
        mean=[20.0],
        scale=[10.0],
        hidden_dim=4,
        labels=["easy", "medium", "hard"],
        actions={"easy": "L8", "medium": "L12", "hard": "L24"},
    ))

    normalized = policy._tensor_from_features({"frame_number": 30.0})

    assert torch.allclose(normalized, torch.tensor([1.0]))


def test_mlp_frame_policy_supports_four_trainable_actions():
    import torch
    from projects.mmdet3d_plugin.models.temporal_memory.policies import build_frame_policy

    policy = build_frame_policy(dict(
        type="mlp",
        checkpoint=None,
        load_weights=False,
        feature_names=["speed"],
        hidden_dim=4,
        labels=["easy", "medium", "hard", "very_hard"],
        actions={
            "easy": "L8",
            "medium": "L12",
            "hard": "L16",
            "very_hard": "L24",
        },
    ))

    logits = policy.logits({"speed": 1.0})

    assert isinstance(policy, torch.nn.Module)
    assert logits.shape == (1, 4)
    assert policy.decide(1, True, policy_features={"speed": 1.0}) in {"L8", "L12", "L16", "L24"}


def test_controller_forces_l24_every_configured_number_of_frames():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        updater="reuse",
        policy=dict(type="fixed_interval", keyframe_interval=99, partial_action="L8"),
        max_frames_between_full=5,
    )

    actions = []
    for _ in range(7):
        actions.append(controller.peek_action(scene_token="scene-a"))
        controller.get_or_update(lambda: "full", scene_token="scene-a")

    assert actions == ["L24", "L8", "L8", "L8", "L8", "L24", "L8"]
    assert controller.policy_stats()["forced_l24"] == 1


def test_controller_passes_policy_features_to_mlp_policy(tmp_path):
    import torch
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    checkpoint = tmp_path / "policy.pt"
    torch.save({
        "feature_names": ["risk"],
        "mean": [0.0],
        "scale": [1.0],
        "hidden_dim": 2,
        "labels": ["easy", "medium", "hard"],
        "actions": {"easy": "L8", "medium": "L12", "hard": "L24"},
        "state_dict": {
            "0.weight": torch.tensor([[1.0], [-1.0]]),
            "0.bias": torch.tensor([0.0, 0.0]),
            "2.weight": torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
            "2.bias": torch.tensor([0.0, 0.0, 0.0]),
        },
    }, checkpoint)
    controller = TemporalMemoryController(
        enabled=True,
        updater="reuse",
        policy=dict(type="mlp", checkpoint=str(checkpoint)),
    )
    controller.get_or_update(lambda: "full", scene_token="scene-a")

    assert controller.peek_action(scene_token="scene-a", policy_features={"risk": 1.0}) == "L8"
    assert controller.get_or_update(
        lambda: "unused",
        scene_token="scene-a",
        policy_features={"risk": -1.0},
    ) == "unused"
    assert controller.last_action == "L24"
