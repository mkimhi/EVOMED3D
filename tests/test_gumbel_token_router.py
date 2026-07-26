import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_gumbel_token_router_preserves_dense_shape_and_tracks_budget():
    from projects.mmdet3d_plugin.models.routing import GumbelTokenRouter

    router = GumbelTokenRouter(
        embed_dims=4,
        hidden_dims=8,
        keep_ratio=0.5,
        temperature=1.0,
        use_global_context=True,
    )
    router.train()

    tokens = torch.randn(2, 6, 4, requires_grad=True)
    routed, metadata = router(tokens)

    assert routed.shape == tokens.shape
    assert metadata['routing_mode'] == 'learned_gumbel_ambient'
    assert metadata['routing_num_tokens'] == 6
    assert metadata['routing_keep_count'] == [3, 3]
    assert metadata['routing_preserves_dense_geometry'] is True

    routed.sum().backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_gumbel_token_router_eval_is_deterministic_topk_with_ambient_fill():
    from projects.mmdet3d_plugin.models.routing import GumbelTokenRouter

    router = GumbelTokenRouter(
        embed_dims=1,
        hidden_dims=1,
        keep_ratio=0.5,
        temperature=1.0,
        use_global_context=False,
    )
    router.eval()
    with torch.no_grad():
        router.scorer[0].weight.fill_(1.0)
        router.scorer[0].bias.zero_()
        router.scorer[2].weight.fill_(1.0)
        router.scorer[2].bias.zero_()

    tokens = torch.tensor([[[1.0], [4.0], [2.0], [3.0]]])
    routed, metadata = router(tokens)

    assert metadata['routing_keep_count'] == [2]
    assert metadata['routing_keep_mask'] == [[False, True, False, True]]
    assert torch.allclose(
        routed,
        torch.tensor([[[1.5], [4.0], [1.5], [3.0]]]),
    )


def test_gumbel_token_router_select_mask_exposes_hard_budget():
    from projects.mmdet3d_plugin.models.routing import GumbelTokenRouter

    router = GumbelTokenRouter(
        embed_dims=1,
        hidden_dims=1,
        keep_ratio=0.5,
        temperature=1.0,
        use_global_context=False,
    )
    router.eval()
    with torch.no_grad():
        router.scorer[0].weight.fill_(1.0)
        router.scorer[0].bias.zero_()
        router.scorer[2].weight.fill_(1.0)
        router.scorer[2].bias.zero_()

    keep_mask, metadata = router.select_mask(torch.tensor([[[1.0], [4.0], [2.0], [3.0]]]))

    assert keep_mask.tolist() == [[False, True, False, True]]
    assert metadata['routing_keep_count'] == [2]
    assert metadata['routing_preserves_dense_geometry'] is False


def test_pack_tokens_with_ambient_reduces_sequence_and_appends_context():
    from projects.mmdet3d_plugin.models.routing import pack_tokens_with_ambient

    tokens = torch.tensor([[[1.0], [4.0], [2.0], [3.0]]])
    pos_embed = torch.tensor([[[10.0], [40.0], [20.0], [30.0]]])
    keep_mask = torch.tensor([[False, True, False, True]])

    packed_tokens, packed_pos, metadata = pack_tokens_with_ambient(
        tokens, pos_embed, keep_mask, ambient_context_len=1)

    assert packed_tokens.shape == (1, 3, 1)
    assert packed_pos.shape == (1, 3, 1)
    assert packed_tokens.tolist() == [[[4.0], [3.0], [1.5]]]
    assert packed_pos.tolist() == [[[40.0], [30.0], [15.0]]]
    assert metadata['routing_num_tokens_after'] == 3
