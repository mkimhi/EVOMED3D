import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_eva_vit_loads_legacy_simfp_keys_into_adapter():
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT

    model = EVAViT(
        img_size=64,
        patch_size=16,
        embed_dim=16,
        depth=1,
        num_heads=2,
        window_size=4,
        global_window_size=4,
        sim_fpn=dict(
            scale_factors=[1],
            in_channels=16,
            out_channels=4,
            out_indices=[4],
        ),
        flash_attn=False,
        with_cp=False,
    )

    key = 'backbone.simfp_4.0.weight'
    target = 'adapter.simfp_4.0.weight'
    legacy_state = {key: torch.full_like(model.state_dict()[target], 7.0)}

    missing, unexpected = model.load_state_dict(legacy_state, strict=False)

    assert key not in unexpected
    assert target not in missing
    assert torch.equal(model.state_dict()[target], legacy_state[key])
