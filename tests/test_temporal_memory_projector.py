import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_partial_feature_projector_returns_repdetr3d_feature_pyramid_shapes():
    from projects.mmdet3d_plugin.models.temporal_memory import PartialFeatureProjector

    projector = PartialFeatureProjector(
        in_channels=8,
        out_channels=4,
        scale_factors=[4, 2, 1, 0.5],
    )

    features = projector(torch.randn(2, 3, 8, 5, 10))

    assert [tuple(feature.shape) for feature in features] == [
        (2, 3, 4, 20, 40),
        (2, 3, 4, 10, 20),
        (2, 3, 4, 5, 10),
        (2, 3, 4, 2, 5),
    ]


def test_partial_feature_projector_rejects_non_spatial_memory():
    from projects.mmdet3d_plugin.models.temporal_memory import PartialFeatureProjector

    projector = PartialFeatureProjector(in_channels=8, out_channels=4)

    with pytest.raises(ValueError, match="B,N_cam,C,H,W"):
        projector(torch.randn(2, 8, 5, 10))


def test_layered_partial_feature_projector_uses_one_projector_per_policy_layer():
    from projects.mmdet3d_plugin.models.temporal_memory import LayeredPartialFeatureProjector

    projector = LayeredPartialFeatureProjector(
        layers=(8, 12, 16),
        projector_cfg=dict(
            in_channels=8,
            out_channels=4,
            scale_factors=[4, 2, 1, 0.5],
            out_indices=[2, 3, 4, 5],
        ),
    )

    out_l8 = projector(torch.randn(2, 3, 8, 5, 10), action="L8")
    out_l12 = projector(torch.randn(2, 3, 8, 5, 10), layer_id="block12")

    assert set(projector.projectors.keys()) == {"8", "12", "16"}
    assert tuple(out_l8[0].shape) == (2, 3, 4, 20, 40)
    assert tuple(out_l12[0].shape) == (2, 3, 4, 20, 40)
