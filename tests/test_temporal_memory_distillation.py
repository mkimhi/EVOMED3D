import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_controller_adds_stop_gradient_teacher_loss_on_training_non_keyframe():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    student = torch.ones(1, 1, 2, 2, requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)

    class Updater:
        def __call__(self, previous_memory, **kwargs):
            return student

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=3,
        updater=Updater(),
        train_with_teacher=True,
        distill_weight=2.0,
    )

    assert controller.get_or_update(lambda: torch.full_like(student, 3), scene_token="s") is not None
    out = controller.get_or_update(
        lambda: torch.full_like(student, 4),
        scene_token="s",
        extract_teacher=lambda: teacher,
        training=True,
    )

    assert out is student
    assert controller.last_distillation_loss is not None
    controller.last_distillation_loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_controller_does_not_call_teacher_when_disabled():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=2,
        updater="reuse",
        train_with_teacher=False,
    )

    controller.get_or_update(lambda: "full", scene_token="s")
    assert controller.get_or_update(
        lambda: "unused",
        scene_token="s",
        extract_teacher=lambda: (_ for _ in ()).throw(AssertionError("teacher called")),
        training=True,
    ) == "full"
    assert controller.last_distillation_loss is None


def test_controller_blocks_teacher_on_inference_non_keyframe_when_forbidden():
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryController

    controller = TemporalMemoryController(
        enabled=True,
        keyframe_interval=2,
        updater="reuse",
        train_with_teacher=True,
        inference_teacher_forbidden=True,
    )

    controller.get_or_update(lambda: "full", scene_token="s")
    with pytest.raises(AssertionError, match="teacher"):
        controller.get_or_update(
            lambda: "unused",
            scene_token="s",
            extract_teacher=lambda: "teacher",
            training=False,
        )
