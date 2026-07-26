import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_relative_ego_transform_identity_is_identity():
    from projects.mmdet3d_plugin.models.temporal_memory import relative_ego_transform

    pose = torch.eye(4)

    assert torch.allclose(relative_ego_transform(pose, pose), torch.eye(4))


def test_transform_points_identity_leaves_points_unchanged():
    from projects.mmdet3d_plugin.models.temporal_memory import transform_points

    points = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])

    assert torch.allclose(transform_points(points, torch.eye(4)), points)
