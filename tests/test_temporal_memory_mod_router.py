import sys
from pathlib import Path

import pytest
import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_mod_capacity_is_exactly_half_per_camera():
    from projects.mmdet3d_plugin.models.temporal_memory.routers import select_topk_tokens

    scores = torch.tensor([
        [0.0, 3.0, 1.0, 2.0],
        [4.0, 1.0, 3.0, 2.0],
    ])
    indices, mask = select_topk_tokens(scores, capacity_ratio=0.5)

    assert indices.tolist() == [[1, 3], [0, 2]]
    assert mask.tolist() == [[False, True, False, True], [True, False, True, False]]


def test_mod_capacity_rounds_up_and_validates_inputs():
    from projects.mmdet3d_plugin.models.temporal_memory.routers import select_topk_tokens

    indices, mask = select_topk_tokens(torch.arange(10.0).reshape(2, 5), 0.5)

    assert indices.shape == (2, 3)
    assert mask.sum(dim=1).tolist() == [3, 3]
    with pytest.raises(ValueError, match="two-dimensional"):
        select_topk_tokens(torch.ones(2, 2, 2), 0.5)
    with pytest.raises(ValueError, match="capacity_ratio"):
        select_topk_tokens(torch.ones(2, 4), 0.0)


def test_indexed_rope_matches_full_rope_at_original_positions():
    from projects.mmdet3d_plugin.models.backbones.eva_vit import VisionRotaryEmbeddingFast

    rope = VisionRotaryEmbeddingFast(dim=2, pt_seq_len=2, ft_seq_len=2)
    dense = torch.arange(32.0).reshape(1, 2, 4, 4)
    indices = torch.tensor([[3, 0]])
    gather_index = indices[:, None, :, None].expand(1, 2, 2, 4)
    selected = dense.gather(2, gather_index)

    expected = rope(dense).gather(2, gather_index)
    actual = rope.forward_indices(selected, indices)

    assert torch.equal(actual, expected)


class _FixedTokenRouter(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = nn.Parameter(torch.as_tensor(scores, dtype=torch.float32))

    def forward(self, tokens):
        return self.scores.expand(tokens.size(0), -1)


def _make_block(window_size=0):
    from projects.mmdet3d_plugin.models.backbones.eva_vit import Block

    return Block(
        dim=8,
        num_heads=2,
        mlp_ratio=1.0,
        window_size=window_size,
        rope=None,
        flash_attn=False,
        with_cp=False,
    )


def test_mod_bypasses_unselected_tokens_and_routes_gradients():
    from projects.mmdet3d_plugin.models.temporal_memory.routers import TokenDepthAdapter

    block = _make_block().train()
    router = _FixedTokenRouter([[0.0, 3.0, 1.0, 2.0]])
    adapter = TokenDepthAdapter(channels=8, hidden_channels=4)
    x = torch.randn(1, 2, 2, 8, requires_grad=True)

    output, stats = block.forward_mod(x, router, adapter, capacity_ratio=0.5)
    output_flat = output.reshape(1, 4, 8)
    input_flat = x.reshape(1, 4, 8)

    assert stats["selected_indices"].tolist() == [[1, 3]]
    assert torch.equal(output_flat[:, [0, 2]], input_flat[:, [0, 2]])
    assert not torch.equal(output_flat[:, [1, 3]], input_flat[:, [1, 3]])
    output.square().mean().backward()
    assert router.scores.grad is not None
    assert torch.isfinite(router.scores.grad).all()
    assert any(parameter.grad is not None for parameter in adapter.parameters())


def test_mod_rejects_window_attention_blocks():
    from projects.mmdet3d_plugin.models.temporal_memory.routers import (
        TokenDepthAdapter,
        TokenDepthRouter,
    )

    with pytest.raises(ValueError, match="global-attention"):
        _make_block(window_size=2).forward_mod(
            torch.randn(1, 2, 2, 8),
            TokenDepthRouter(channels=8, hidden_channels=4),
            TokenDepthAdapter(channels=8, hidden_channels=4),
            capacity_ratio=0.5,
        )


def _make_eva():
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT

    return EVAViT(
        img_size=32,
        patch_size=16,
        embed_dim=8,
        depth=24,
        num_heads=2,
        mlp_ratio=1.0,
        window_size=2,
        global_window_size=0,
        window_block_indexes=tuple(
            index for index in range(24) if index not in (8, 11, 14, 17, 20, 23)
        ),
        sim_fpn=None,
        flash_attn=False,
        with_cp=False,
    )


def test_eva_temporal_mod_routes_only_configured_global_blocks():
    backbone = _make_eva().train()
    backbone.configure_temporal_mod(
        routed_block_indices=(8, 11, 14, 17, 20),
        capacity_ratio=0.5,
        router_hidden_channels=4,
        adapter_hidden_channels=4,
    )

    output = backbone(torch.randn(2, 3, 32, 32))[0]
    stats = backbone.get_temporal_mod_stats()

    assert output.shape == (2, 8, 2, 2)
    assert tuple(stats) == (8, 11, 14, 17, 20)
    assert all(item["selected_count"] == 2 for item in stats.values())
    assert all(item["token_count"] == 4 for item in stats.values())
    output.square().mean().backward()
    assert all(
        any(parameter.grad is not None for parameter in router.parameters())
        for router in backbone.temporal_mod_routers.values()
    )


def test_eva_dense_path_is_unchanged_when_temporal_mod_is_disabled():
    backbone = _make_eva().eval()
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = backbone(images)[0]
        backbone.configure_temporal_mod(
            routed_block_indices=(8, 11, 14, 17, 20),
            capacity_ratio=0.5,
            router_hidden_channels=4,
            adapter_hidden_channels=4,
        )
        backbone.clear_temporal_mod()
        actual = backbone(images)[0]

    assert torch.equal(actual, expected)


def test_eva_temporal_mod_rejects_non_global_and_final_blocks():
    backbone = _make_eva()

    with pytest.raises(ValueError, match="global-attention"):
        backbone.configure_temporal_mod(routed_block_indices=(7,))
    with pytest.raises(ValueError, match="final block"):
        backbone.configure_temporal_mod(routed_block_indices=(23,))
