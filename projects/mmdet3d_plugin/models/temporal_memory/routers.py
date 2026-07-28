from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DepthRouteOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    hard_actions: torch.Tensor
    hard_one_hot: torch.Tensor
    expected_cost: torch.Tensor


def _pool_camera_spatial(features):
    if features is None:
        return None
    if features.dim() != 5:
        raise ValueError('router features must be [B, N_cam, C, H, W]')
    return features.mean(dim=(1, 3, 4))


class DepthExitRouter(nn.Module):
    actions = ('L8', 'L12', 'L16', 'L24')

    def __init__(
        self,
        feature_dim,
        metadata_dim,
        hidden_dim=128,
        temperature=1.0,
        costs=(8 / 24, 12 / 24, 16 / 24, 1.0),
    ):
        super().__init__()
        if len(costs) != len(self.actions):
            raise ValueError('costs must match the number of depth actions')
        self.current_proj = nn.Linear(feature_dim, hidden_dim)
        self.previous_proj = nn.Linear(feature_dim, hidden_dim)
        self.metadata_proj = nn.Linear(metadata_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.actions)),
        )
        self.temperature = float(temperature)
        self.register_buffer('costs', torch.as_tensor(costs, dtype=torch.float32))

    def forward(self, current_l8, previous_memory, metadata, training=None):
        current = _pool_camera_spatial(current_l8)
        if previous_memory is None:
            previous = current.new_zeros(current.shape)
        else:
            previous = _pool_camera_spatial(previous_memory)
            if previous.shape != current.shape:
                raise ValueError('previous memory pool must match current L8 pool')
        if metadata.dim() != 2 or metadata.size(0) != current.size(0):
            raise ValueError('metadata must be [B, metadata_dim] matching current L8 batch')

        hidden = torch.cat((
            self.current_proj(current),
            self.previous_proj(previous),
            self.metadata_proj(metadata.to(dtype=current.dtype)),
        ), dim=-1)
        logits = self.classifier(hidden)
        probabilities = logits.softmax(dim=-1)
        use_training = self.training if training is None else bool(training)
        if use_training:
            hard_one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        else:
            hard_one_hot = F.one_hot(logits.argmax(dim=-1), num_classes=len(self.actions)).to(logits.dtype)
        hard_actions = hard_one_hot.argmax(dim=-1)
        expected_cost = (probabilities * self.costs.to(probabilities)).sum(dim=-1).mean()
        return DepthRouteOutput(
            logits=logits,
            probabilities=probabilities,
            hard_actions=hard_actions,
            hard_one_hot=hard_one_hot,
            expected_cost=expected_cost,
        )


def depth_router_losses(output, target_cost=0.5, balance_prior=None, balance_weight=0.0):
    budget = (output.expected_cost - float(target_cost)).square()
    balance = output.expected_cost.new_zeros(())
    if balance_weight:
        if balance_prior is None:
            balance_prior = [1.0 / output.probabilities.size(-1)] * output.probabilities.size(-1)
        prior = torch.as_tensor(
            balance_prior,
            device=output.probabilities.device,
            dtype=output.probabilities.dtype,
        )
        if prior.numel() != output.probabilities.size(-1):
            raise ValueError('balance_prior must match the number of depth actions')
        balance = (output.probabilities.mean(dim=0) - prior).square().mean() * float(balance_weight)
    return {
        'loss_tm_budget': budget,
        'loss_tm_balance': balance,
        'expected_cost': output.expected_cost.detach(),
    }
