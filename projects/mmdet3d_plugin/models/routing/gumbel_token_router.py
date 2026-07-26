import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class GumbelTokenRouter(nn.Module):
    """Learned dense token router with Gumbel top-k selection.

    The output keeps the same [B, N, C] shape as the input so downstream dense
    spatial indexing remains valid. Dropped token positions are replaced by a
    pooled ambient context vector.
    """

    def __init__(
        self,
        embed_dims: int,
        hidden_dims: Optional[int] = None,
        keep_ratio: float = 0.5,
        temperature: float = 1.0,
        use_global_context: bool = True,
    ):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.embed_dims = int(embed_dims)
        self.keep_ratio = float(keep_ratio)
        self.temperature = float(temperature)
        self.use_global_context = bool(use_global_context)

        hidden_dims = int(hidden_dims or embed_dims)
        scorer_in_dims = embed_dims * 2 if use_global_context else embed_dims
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        keep_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, N, C].")

        keep_mask, metadata = self.select_mask(tokens, keep_ratio)
        logits = metadata.pop("_routing_logits")
        gate = self._make_straight_through_gate(logits, keep_mask)
        ambient = self._ambient_context(tokens, keep_mask)
        routed = tokens * gate.unsqueeze(-1) + ambient * (1.0 - gate).unsqueeze(-1)

        metadata["routing_preserves_dense_geometry"] = True
        return routed, metadata

    def select_mask(
        self,
        tokens: torch.Tensor,
        keep_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, N, C].")

        logits = self._score(tokens)
        target_ratio = self._target_ratio(logits, keep_ratio)
        keep_mask = self._make_keep_mask(logits, target_ratio)
        keep_count = keep_mask.sum(dim=1)
        metadata = {
            "routing_mode": "learned_gumbel_ambient",
            "routing_num_tokens": int(tokens.shape[1]),
            "routing_keep_count": keep_count.detach().cpu().tolist(),
            "routing_keep_ratio_actual": (keep_count.float() / tokens.shape[1])
            .detach()
            .cpu()
            .tolist(),
            "routing_target_keep_ratio": target_ratio.detach().cpu().tolist(),
            "routing_preserves_dense_geometry": False,
            "routing_keep_mask": keep_mask.detach().cpu().tolist(),
            "_routing_logits": logits,
        }
        return keep_mask, metadata

    def _score(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_global_context:
            global_context = tokens.mean(dim=1, keepdim=True).expand_as(tokens)
            tokens = torch.cat([tokens, global_context], dim=-1)
        return self.scorer(tokens).squeeze(-1)

    def _target_ratio(
        self,
        logits: torch.Tensor,
        keep_ratio: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if keep_ratio is None:
            ratio = logits.new_full((logits.shape[0],), self.keep_ratio)
        else:
            ratio = torch.as_tensor(
                keep_ratio, device=logits.device, dtype=logits.dtype
            ).reshape(-1)
            if ratio.numel() == 1:
                ratio = ratio.expand(logits.shape[0])
            elif ratio.numel() != logits.shape[0] and logits.shape[0] % ratio.numel() == 0:
                ratio = ratio.repeat_interleave(logits.shape[0] // ratio.numel())
            if ratio.numel() != logits.shape[0]:
                raise ValueError(
                    f"keep_ratio must have 1, {logits.shape[0]}, or an even "
                    f"divisor of {logits.shape[0]} values, got {ratio.numel()}"
                )
        return ratio.clamp(1.0 / logits.shape[1], 1.0)

    def _make_keep_mask(
        self,
        logits: torch.Tensor,
        target_ratio: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            logits = logits + self._sample_gumbel(logits)

        num_tokens = logits.shape[1]
        keep_counts = torch.ceil(target_ratio * num_tokens).long().clamp(1, num_tokens)
        sorted_idx = torch.argsort(logits, dim=1, descending=True)
        keep_mask = torch.zeros_like(logits, dtype=torch.bool)
        for batch_idx, keep_count in enumerate(keep_counts.tolist()):
            keep_mask[batch_idx, sorted_idx[batch_idx, :keep_count]] = True
        return keep_mask

    def _make_straight_through_gate(
        self,
        logits: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        hard_gate = keep_mask.to(dtype=logits.dtype)
        if not self.training:
            return hard_gate

        soft_gate = torch.sigmoid(logits / self.temperature)
        return hard_gate + soft_gate - soft_gate.detach()

    def _ambient_context(
        self,
        tokens: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        drop_mask = (~keep_mask).unsqueeze(-1).to(dtype=tokens.dtype)
        drop_count = drop_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ambient = (tokens * drop_mask).sum(dim=1, keepdim=True) / drop_count
        all_kept = keep_mask.all(dim=1).view(-1, 1, 1)
        fallback = tokens.mean(dim=1, keepdim=True)
        return torch.where(all_kept, fallback, ambient)

    def _sample_gumbel(self, logits: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(logits.dtype).eps
        uniform = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
        return -torch.log(-torch.log(uniform)) * self.temperature


def pack_tokens_with_ambient(
    tokens: torch.Tensor,
    pos_embed: torch.Tensor,
    keep_mask: torch.Tensor,
    ambient_context_len: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Pack kept tokens for non-deformable attention and append ambient context."""
    if tokens.ndim != 3 or pos_embed.ndim != 3:
        raise ValueError("tokens and pos_embed must have shape [B, N, C].")
    if tokens.shape[:2] != pos_embed.shape[:2]:
        raise ValueError("tokens and pos_embed must share batch and token dimensions.")
    if keep_mask.shape != tokens.shape[:2]:
        raise ValueError("keep_mask must have shape [B, N].")
    if ambient_context_len not in (0, 1):
        raise ValueError("ambient_context_len currently supports only 0 or 1.")

    keep_counts = keep_mask.sum(dim=1)
    if not torch.equal(keep_counts, keep_counts.new_full(keep_counts.shape, keep_counts[0])):
        raise ValueError("Packed routing requires the same keep count for every batch item.")

    batch_size, num_tokens, _ = tokens.shape
    keep_count = int(keep_counts[0].item())
    gather_idx = torch.arange(num_tokens, device=tokens.device).expand(batch_size, -1)
    gather_idx = gather_idx[keep_mask].view(batch_size, keep_count)
    token_idx = gather_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    pos_idx = gather_idx.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])
    packed_tokens = torch.gather(tokens, 1, token_idx)
    packed_pos = torch.gather(pos_embed, 1, pos_idx)

    if ambient_context_len == 1 and keep_count < num_tokens:
        drop_mask = (~keep_mask).unsqueeze(-1).to(dtype=tokens.dtype)
        drop_count = drop_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ambient_tokens = (tokens * drop_mask).sum(dim=1, keepdim=True) / drop_count
        ambient_pos = (pos_embed * drop_mask).sum(dim=1, keepdim=True) / drop_count
        packed_tokens = torch.cat([packed_tokens, ambient_tokens], dim=1)
        packed_pos = torch.cat([packed_pos, ambient_pos], dim=1)

    metadata = {
        "routing_num_tokens_before": int(num_tokens),
        "routing_num_tokens_after": int(packed_tokens.shape[1]),
        "routing_ambient_context_len": int(packed_tokens.shape[1] - keep_count),
    }
    return packed_tokens, packed_pos, metadata
