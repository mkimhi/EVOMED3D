import importlib.util
from pathlib import Path

import torch


ROUTER_PATH = (
    Path(__file__).resolve().parents[3]
    / "projects"
    / "mmdet3d_plugin"
    / "models"
    / "routing"
    / "tri_state_router.py"
)


def _load_router():
    spec = importlib.util.spec_from_file_location("tri_state_router", ROUTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tri_state_router_tensor_utilities():
    router = _load_router()
    tokens = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )
    keep_mask = torch.tensor([True, False, True, False])
    bypass_mask = torch.tensor([False, True, False, False])

    final_mask = router.combine_keep_and_bypass(keep_mask, bypass_mask)
    expected_mask = torch.tensor([True, True, True, False])
    assert torch.equal(final_mask, expected_mask)

    hard_drop = router.route_tokens(tokens, final_mask, mode="hard_drop")
    assert hard_drop.shape == torch.Size([3, 2])
    assert torch.equal(hard_drop, tokens[expected_mask])

    ambient = router.route_tokens(tokens, final_mask, mode="ambient_pool")
    assert ambient.shape == torch.Size([4, 2])
    assert torch.equal(ambient[:-1], tokens[expected_mask])

    ambient_token = router.pool_ambient_tokens(tokens, final_mask)
    expected_ambient = tokens[~expected_mask].mean(dim=0, keepdim=True)
    assert torch.allclose(ambient_token, expected_ambient)
    assert torch.allclose(ambient[-1:], expected_ambient)


def test_tri_state_router_batched_padding_mask():
    router = _load_router()
    tokens = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ]
    )
    final_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )

    routed, key_padding_mask = router.route_tokens(
        tokens, final_mask, mode="ambient_pool", return_key_padding_mask=True
    )

    assert routed.shape == torch.Size([2, 3, 2])
    assert key_padding_mask.shape == torch.Size([2, 3])
    assert torch.equal(key_padding_mask[0], torch.tensor([False, False, False]))
    assert torch.equal(key_padding_mask[1], torch.tensor([False, False, True]))
    assert torch.allclose(routed[1, 1], tokens[1, ~final_mask[1]].mean(dim=0))
    assert torch.equal(routed[1, 2], torch.zeros_like(routed[1, 2]))


def test_make_budget_mask_uses_speed_conditioned_keep_ratios():
    router = _load_router()
    scores = torch.tensor(
        [
            [1.0, 4.0, 3.0, 2.0],
            [10.0, 1.0, 9.0, 8.0],
            [1.0, 3.0, 2.0, 4.0],
        ]
    )
    ego_speed = torch.tensor([0.0, 5.0, 20.0])

    mask, target_ratio = router.make_budget_mask(
        scores,
        default_keep_ratio=0.75,
        ego_speed=ego_speed,
        stopped_keep_ratio=0.25,
        moderate_keep_ratio=0.5,
        fast_keep_ratio=0.75,
        stopped_speed=0.5,
        high_speed=12.0,
    )

    assert torch.allclose(target_ratio, torch.tensor([0.25, 0.5, 0.75]))
    assert mask.sum(dim=1).tolist() == [1, 2, 3]
    assert torch.equal(mask[0], torch.tensor([False, True, False, False]))
    assert torch.equal(mask[1], torch.tensor([True, False, True, False]))
    assert torch.equal(mask[2], torch.tensor([False, True, True, True]))



def test_make_budget_mask_repeats_batch_speed_for_camera_rows():
    router = _load_router()
    scores = torch.tensor(
        [
            [1.0, 4.0, 3.0, 2.0],
            [4.0, 1.0, 3.0, 2.0],
            [10.0, 1.0, 9.0, 8.0],
            [1.0, 10.0, 9.0, 8.0],
        ]
    )
    ego_speed = torch.tensor([0.0, 20.0])

    mask, target_ratio = router.make_budget_mask(
        scores,
        default_keep_ratio=0.5,
        ego_speed=ego_speed,
        stopped_keep_ratio=0.25,
        fast_keep_ratio=0.75,
        stopped_speed=0.5,
        high_speed=12.0,
    )

    assert torch.allclose(target_ratio, torch.tensor([0.25, 0.25, 0.75, 0.75]))
    assert mask.sum(dim=1).tolist() == [1, 1, 3, 3]

def test_apply_dense_budget_mask_preserves_shape_and_pools_dropped_tokens():
    router = _load_router()
    tokens = torch.tensor(
        [
            [[1.0, 1.0], [10.0, 10.0], [3.0, 3.0], [2.0, 2.0]],
        ]
    )
    keep_mask = torch.tensor([[False, True, True, False]])

    masked, metadata = router.apply_dense_budget_mask(
        tokens,
        keep_mask,
        mode="ambient_pool",
    )

    expected_ambient = torch.tensor([[[1.5, 1.5]]])
    assert masked.shape == tokens.shape
    assert torch.equal(masked[:, 1:3], tokens[:, 1:3])
    assert torch.allclose(masked[:, 0:1], expected_ambient)
    assert torch.allclose(masked[:, 3:4], expected_ambient)
    assert metadata["routing_keep_count"] == [2]
    assert metadata["routing_num_tokens"] == 4
