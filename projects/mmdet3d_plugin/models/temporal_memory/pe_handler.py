import torch


def relative_ego_transform(global_ego_prev, global_ego_curr):
    """Return transform from previous ego coordinates into current ego coordinates."""
    return torch.linalg.inv(global_ego_curr) @ global_ego_prev


def transform_points(points, transform):
    ones = torch.ones_like(points[..., :1])
    points_h = torch.cat([points, ones], dim=-1)
    transformed = points_h @ transform.transpose(-1, -2)
    return transformed[..., :3]
