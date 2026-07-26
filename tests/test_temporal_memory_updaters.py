import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_gated_residual_updater_starts_as_exact_reuse_for_tensor():
    from projects.mmdet3d_plugin.models.temporal_memory import GatedResidualMemoryUpdater

    updater = GatedResidualMemoryUpdater(channels=4, ego_dim=3)
    previous = torch.randn(2, 6, 4, 3, 5)
    partial = torch.randn_like(previous)
    ego = torch.randn(2, 3)

    out = updater(previous, partial, ego)

    assert torch.equal(out, previous)


def test_gated_residual_updater_preserves_shape_and_gradients_for_tokens():
    from projects.mmdet3d_plugin.models.temporal_memory import GatedResidualMemoryUpdater

    updater = GatedResidualMemoryUpdater(channels=4, ego_dim=2, feature_format="tokens")
    updater.residual_scale.data.fill_(0.1)
    previous = torch.randn(2, 6, 9, 4, requires_grad=True)
    partial = torch.randn_like(previous)
    ego = torch.randn(2, 2)

    out = updater(previous, partial, ego)
    out.sum().backward()

    assert out.shape == previous.shape
    assert previous.grad is not None
    assert updater.residual.weight.grad is not None


def test_gated_residual_updater_rejects_missing_partial_features():
    from projects.mmdet3d_plugin.models.temporal_memory import GatedResidualMemoryUpdater

    with pytest.raises(ValueError, match="partial_current"):
        GatedResidualMemoryUpdater(channels=4)(torch.randn(2, 6, 4, 3, 5))


def test_partial_only_updater_returns_projected_current_features():
    from projects.mmdet3d_plugin.models.temporal_memory import build_memory_updater

    updater = build_memory_updater("partial_only")
    previous = [torch.randn(2, 6, 4, 3, 5)]
    partial = [torch.randn(2, 6, 4, 3, 5)]

    assert updater(previous, partial_current=partial) is partial


def test_partial_only_updater_rejects_missing_partial_features():
    from projects.mmdet3d_plugin.models.temporal_memory import build_memory_updater

    updater = build_memory_updater("partial_only")

    with pytest.raises(ValueError, match="partial_current"):
        updater(torch.randn(2, 6, 4, 3, 5))


def test_attention_memory_updater_starts_as_exact_reuse_for_list():
    from projects.mmdet3d_plugin.models.temporal_memory import build_memory_updater

    updater = build_memory_updater(
        "attention_residual",
        channels=4,
        num_heads=2,
        kv_pool_size=(2, 2),
    )
    previous = [torch.randn(2, 3, 4, 5, 7), torch.randn(2, 3, 4, 3, 4)]
    partial = [torch.randn_like(item) for item in previous]

    out = updater(previous, partial_current=partial)

    assert isinstance(out, list)
    for output, expected in zip(out, previous):
        assert torch.equal(output, expected)


@pytest.mark.parametrize(
    "name",
    [
        "attention_residual",
        "gated_attention_residual",
        "per_camera_attention_residual",
        "inverted_attention_residual",
    ],
)
def test_attention_memory_updater_preserves_shape_and_gradients(name):
    from projects.mmdet3d_plugin.models.temporal_memory import build_memory_updater

    updater = build_memory_updater(
        name,
        channels=4,
        num_heads=2,
        kv_pool_size=(2, 2),
    )
    updater.residual_scale.data.fill_(0.1)
    previous = torch.randn(2, 3, 4, 5, 7, requires_grad=True)
    partial = torch.randn_like(previous)

    out = updater(previous, partial_current=partial)
    out.sum().backward()

    assert out.shape == previous.shape
    assert previous.grad is not None
    assert updater.attn.in_proj_weight.grad is not None


def test_attention_memory_updater_rejects_missing_partial_features():
    from projects.mmdet3d_plugin.models.temporal_memory import build_memory_updater

    updater = build_memory_updater("attention_residual", channels=4, num_heads=2)

    with pytest.raises(ValueError, match="partial_current"):
        updater(torch.randn(2, 6, 4, 3, 5))


def test_layered_gated_residual_updater_can_share_gate_across_policy_layers():
    from projects.mmdet3d_plugin.models.temporal_memory import build_memory_updater

    updater = build_memory_updater(
        "gated_residual",
        channels=4,
        layers=(8, 12, 16),
        shared_memory_gate=True,
    )
    previous = [torch.randn(2, 3, 4, 5, 7)]
    partial = [torch.randn_like(previous[0])]

    out_l8 = updater(previous, partial_current=partial, action="L8")
    out_l16 = updater(previous, partial_current=partial, action="L16")

    assert updater.updaters["8"] is updater.updaters["16"]
    assert torch.equal(out_l8[0], previous[0])
    assert torch.equal(out_l16[0], previous[0])


def test_layered_gated_residual_updater_can_use_per_layer_gates():
    from projects.mmdet3d_plugin.models.temporal_memory import build_memory_updater

    updater = build_memory_updater(
        "gated_residual",
        channels=4,
        layers=(8, 12),
        shared_memory_gate=False,
    )

    assert updater.updaters["8"] is not updater.updaters["12"]
