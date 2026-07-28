import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_eva_vit_returns_one_requested_intermediate_feature():
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT

    backbone = EVAViT(
        img_size=64,
        patch_size=16,
        embed_dim=16,
        depth=4,
        num_heads=2,
        window_size=4,
        global_window_size=4,
        sim_fpn=None,
        flash_attn=False,
        with_cp=False,
    )
    backbone.eval()

    with torch.no_grad():
        feature = backbone.forward_intermediate(torch.randn(2, 3, 64, 64), layer_index=1)

    assert tuple(feature.shape) == (2, 16, 4, 4)


@pytest.mark.parametrize("stop_at", [1, 3])
def test_eva_vit_continuation_matches_direct_prefix(stop_at):
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT

    backbone = EVAViT(
        img_size=64,
        patch_size=16,
        embed_dim=16,
        depth=4,
        num_heads=2,
        window_size=4,
        global_window_size=4,
        sim_fpn=None,
        flash_attn=False,
        with_cp=False,
    ).eval()
    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        tokens = backbone.prepare_tokens(images)
        first = backbone.forward_blocks(tokens, start_index=0, stop_at=0)
        continued = backbone.forward_blocks(first, start_index=1, stop_at=stop_at)
        direct = backbone.forward_blocks(tokens, start_index=0, stop_at=stop_at)

    assert torch.allclose(continued, direct, atol=1e-5, rtol=1e-5)


def test_repdetr3d_adapter_extracts_requested_partial_eva_layer():
    from projects.mmdet3d_plugin.models.temporal_memory import RepDetr3DAdapter

    class Backbone:
        def forward_intermediate(self, img, layer_index):
            self.args = (img, layer_index)
            return torch.randn(img.size(0), 8, 4, 5)

    class Model:
        use_grid_mask = False
        img_backbone = Backbone()

    img = torch.randn(2, 3, 3, 32, 40)
    features = RepDetr3DAdapter(debug_shapes=True).extract_partial_features(
        Model(),
        {"img": img},
        layer_id="50%",
    )

    assert tuple(features.shape) == (2, 3, 8, 4, 5)
    flattened_img, layer_index = Model.img_backbone.args
    assert torch.equal(flattened_img, img.reshape(6, 3, 32, 40))
    assert layer_index == 11


def test_repdetr3d_adapter_preserves_sequence_shape_for_partial_eva_layer():
    from projects.mmdet3d_plugin.models.temporal_memory import RepDetr3DAdapter

    class Backbone:
        def forward_intermediate(self, img, layer_index):
            return torch.randn(img.size(0), 8, 4, 5)

    class Model:
        use_grid_mask = False
        img_backbone = Backbone()

    img = torch.randn(2, 4, 3, 3, 32, 40)
    features = RepDetr3DAdapter(debug_shapes=True).extract_partial_features(
        Model(),
        {"img": img},
        layer_id="75%",
    )

    assert tuple(features.shape) == (2, 4, 3, 8, 4, 5)


@pytest.mark.parametrize(
    "layer_id,expected",
    [
        ("25%", 5),
        ("50%", 11),
        ("75%", 17),
        ("quarter", 5),
        ("middle", 11),
        ("three_quarter", 17),
    ],
)
def test_repdetr3d_adapter_resolves_eva02_depth_presets(layer_id, expected):
    from projects.mmdet3d_plugin.models.temporal_memory import resolve_eva_layer_index

    assert resolve_eva_layer_index(layer_id, depth=24) == expected


def test_repdetr3d_adapter_rejects_invalid_partial_feature_shape():
    from projects.mmdet3d_plugin.models.temporal_memory import RepDetr3DAdapter

    class Backbone:
        def forward_intermediate(self, img, layer_index):
            return torch.randn(img.size(0), 4, 5)

    class Model:
        use_grid_mask = False
        img_backbone = Backbone()

    with pytest.raises(AssertionError, match="4D"):
        RepDetr3DAdapter(debug_shapes=True).extract_partial_features(
            Model(),
            {"img": torch.randn(1, 2, 3, 32, 40)},
            layer_id="25%",
        )
