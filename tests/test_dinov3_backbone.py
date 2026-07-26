import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_dinov3_vit_backbone_returns_fpn_features():
    from projects.mmdet3d_plugin.models.backbones import DINOv3ViT

    backbone = DINOv3ViT(
        img_size=(64, 160),
        patch_size=16,
        embed_dim=384,
        depth=1,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=4,
        sim_fpn=dict(
            scale_factors=[4, 2, 1, 0.5],
            in_channels=384,
            out_channels=256,
            out_indices=[2, 3, 4, 5],
        ),
    )
    backbone.eval()

    with torch.no_grad():
        outputs = backbone(torch.randn(2, 3, 64, 160))

    assert len(outputs) == 4
    assert [tuple(out.shape) for out in outputs] == [
        (2, 256, 16, 40),
        (2, 256, 8, 20),
        (2, 256, 4, 10),
        (2, 256, 2, 5),
    ]


def test_dinov3_vit_backbone_supports_standard_mlp_weights():
    from projects.mmdet3d_plugin.models.backbones import DINOv3ViT

    backbone = DINOv3ViT(
        img_size=(64, 160),
        patch_size=16,
        embed_dim=768,
        depth=1,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=4,
        mlp_type='mlp',
        sim_fpn=None,
    )

    state_dict = backbone.state_dict()

    assert 'blocks.0.mlp.fc1.weight' in state_dict
    assert 'blocks.0.mlp.fc1.bias' in state_dict
    assert 'blocks.0.mlp.fc1_g.weight' not in state_dict
    assert tuple(state_dict['blocks.0.mlp.fc1.weight'].shape) == (3072, 768)
