import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_depth_exit_router_selects_one_action_per_sample_and_has_budget_gradients():
    from projects.mmdet3d_plugin.models.temporal_memory.routers import (
        DepthExitRouter,
        depth_router_losses,
    )

    router = DepthExitRouter(feature_dim=4, metadata_dim=3, hidden_dim=8)
    router.train()
    output = router(
        current_l8=torch.randn(4, 6, 4, 2, 2),
        previous_memory=torch.randn(4, 6, 4, 2, 2),
        metadata=torch.randn(4, 3),
    )

    assert output.hard_actions.shape == (4,)
    assert output.hard_one_hot.shape == (4, 4)
    assert output.hard_one_hot.sum(dim=1).tolist() == [1.0, 1.0, 1.0, 1.0]

    losses = depth_router_losses(
        output,
        target_cost=0.5,
        balance_prior=(0.25, 0.25, 0.25, 0.25),
        balance_weight=0.1,
    )
    (losses['loss_tm_budget'] + losses['loss_tm_balance']).backward()

    assert any(parameter.grad is not None for parameter in router.parameters())


def test_depth_exit_router_eval_selects_deterministic_argmax_and_handles_missing_memory():
    from projects.mmdet3d_plugin.models.temporal_memory.routers import DepthExitRouter

    router = DepthExitRouter(feature_dim=2, metadata_dim=1, hidden_dim=4)
    router.eval()
    with torch.no_grad():
        output = router(
            current_l8=torch.ones(2, 6, 2, 1, 1),
            previous_memory=None,
            metadata=torch.zeros(2, 1),
        )

    assert output.hard_actions.tolist() == output.logits.argmax(dim=-1).tolist()
    assert torch.isfinite(output.probabilities).all()
