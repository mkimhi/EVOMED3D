"""Tensor helpers for fixed-token tri-state routing.

These helpers are intentionally independent from MMDetection registries so they
can be tested without importing the full plugin package.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _as_bool_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    if mask.shape != reference.shape[:-1]:
        raise ValueError(
            "Mask shape must match token shape without feature dimension: "
            f"expected {reference.shape[:-1]}, got {mask.shape}"
        )
    return mask


def _pad_routed_sequences(
    sequences: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Expected at least one routed sequence.")

    max_len = max(seq.shape[0] for seq in sequences)
    feature_dim = sequences[0].shape[-1]
    padded = []
    key_padding_masks = []
    for seq in sequences:
        if seq.shape[-1] != feature_dim:
            raise ValueError("All routed sequences must share the same feature dimension.")
        seq_len = seq.shape[0]
        if seq_len < max_len:
            pad = seq.new_zeros((max_len - seq_len, feature_dim))
            seq = torch.cat([seq, pad], dim=0)
        padded.append(seq)

        mask = seq.new_zeros((max_len,), dtype=torch.bool)
        mask[seq_len:] = True
        key_padding_masks.append(mask)

    return torch.stack(padded, dim=0), torch.stack(key_padding_masks, dim=0)


def combine_keep_and_bypass(
    keep_mask: torch.Tensor, bypass_mask: torch.Tensor
) -> torch.Tensor:
    """Return the boolean union of keep and bypass masks."""

    if keep_mask.shape != bypass_mask.shape:
        raise ValueError(
            "keep_mask and bypass_mask must have the same shape: "
            f"{keep_mask.shape} vs {bypass_mask.shape}"
        )
    return keep_mask.to(dtype=torch.bool) | bypass_mask.to(dtype=torch.bool)


def pool_ambient_tokens(
    tokens: torch.Tensor,
    final_mask: torch.Tensor,
    ambient_context_len: int = 1,
) -> torch.Tensor:
    """Pool dropped tokens into fixed-length ambient context tokens."""

    if tokens.ndim not in (2, 3):
        raise ValueError("tokens must have shape [N, C] or [B, N, C].")
    if ambient_context_len < 1:
        raise ValueError("ambient_context_len must be >= 1.")

    final_mask = _as_bool_mask(final_mask, tokens)

    def _pool(sample_tokens: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        dropped_tokens = sample_tokens[~sample_mask]
        if dropped_tokens.numel() == 0:
            return sample_tokens.new_zeros((ambient_context_len, sample_tokens.shape[-1]))
        if ambient_context_len == 1:
            return dropped_tokens.mean(dim=0, keepdim=True)
        pooled = F.adaptive_avg_pool1d(
            dropped_tokens.transpose(0, 1).unsqueeze(0),
            ambient_context_len,
        )
        return pooled.squeeze(0).transpose(0, 1)

    if tokens.ndim == 2:
        return _pool(tokens, final_mask)

    ambient_tokens = [
        _pool(sample_tokens, sample_mask)
        for sample_tokens, sample_mask in zip(tokens, final_mask)
    ]
    return torch.stack(ambient_tokens, dim=0)


def route_tokens(
    tokens: torch.Tensor,
    final_mask: torch.Tensor,
    mode: str = "hard_drop",
    return_key_padding_mask: bool = False,
    return_positions: bool = False,
    ambient_context_len: int = 1,
):
    """Route tokens according to a keep mask.

    Batched output is padded to the longest routed sample.
    """

    if tokens.ndim not in (2, 3):
        raise ValueError("tokens must have shape [N, C] or [B, N, C].")
    if mode not in {"hard_drop", "ambient_pool"}:
        raise ValueError("mode must be either 'hard_drop' or 'ambient_pool'.")
    if ambient_context_len < 1:
        raise ValueError("ambient_context_len must be >= 1.")

    final_mask = _as_bool_mask(final_mask, tokens)

    if tokens.ndim == 2:
        kept_tokens = tokens[final_mask]
        keep_indices = torch.nonzero(final_mask, as_tuple=False).flatten().to(tokens.device)
        if mode == "hard_drop":
            routed = kept_tokens
            positions = keep_indices
        else:
            ambient = pool_ambient_tokens(
                tokens, final_mask, ambient_context_len=ambient_context_len
            )
            routed = torch.cat([kept_tokens, ambient], dim=0)
            positions = torch.cat(
                [keep_indices, keep_indices.new_full((ambient_context_len,), -1)],
                dim=0,
            )
        if return_key_padding_mask:
            key_padding_mask = routed.new_zeros((routed.shape[0],), dtype=torch.bool)
            if return_positions:
                return routed, key_padding_mask, positions
            return routed, key_padding_mask
        if return_positions:
            return routed, positions
        return routed

    routed_sequences = []
    routed_positions = []
    for sample_tokens, sample_mask in zip(tokens, final_mask):
        kept_tokens = sample_tokens[sample_mask]
        keep_indices = torch.nonzero(sample_mask, as_tuple=False).flatten().to(tokens.device)
        if mode == "ambient_pool":
            ambient = pool_ambient_tokens(
                sample_tokens, sample_mask, ambient_context_len=ambient_context_len
            )
            kept_tokens = torch.cat([kept_tokens, ambient], dim=0)
            keep_indices = torch.cat(
                [keep_indices, keep_indices.new_full((ambient_context_len,), -1)],
                dim=0,
            )
        routed_sequences.append(kept_tokens)
        routed_positions.append(keep_indices)

    routed, key_padding_mask = _pad_routed_sequences(routed_sequences)
    max_len = routed.shape[1]
    padded_positions = []
    for positions in routed_positions:
        if positions.shape[0] < max_len:
            pad = positions.new_full((max_len - positions.shape[0],), -1)
            positions = torch.cat([positions, pad], dim=0)
        padded_positions.append(positions)
    routed_positions = torch.stack(padded_positions, dim=0)

    if return_key_padding_mask:
        if return_positions:
            return routed, key_padding_mask, routed_positions
        return routed, key_padding_mask
    if return_positions:
        return routed, routed_positions
    return routed


def _target_keep_ratio(
    scores: torch.Tensor,
    default_keep_ratio: float,
    ego_speed: Optional[torch.Tensor] = None,
    stopped_keep_ratio: Optional[float] = None,
    moderate_keep_ratio: Optional[float] = None,
    fast_keep_ratio: Optional[float] = None,
    stopped_speed: float = 0.5,
    high_speed: float = 12.0,
) -> torch.Tensor:
    ratio = scores.new_full((scores.shape[0],), float(default_keep_ratio))
    if ego_speed is None:
        return ratio.clamp(0.0, 1.0)

    speed = torch.as_tensor(ego_speed, device=scores.device, dtype=scores.dtype).reshape(-1)
    if speed.numel() == 1:
        speed = speed.expand(scores.shape[0])
    elif speed.numel() != scores.shape[0] and scores.shape[0] % speed.numel() == 0:
        speed = speed.repeat_interleave(scores.shape[0] // speed.numel())
    if speed.numel() != scores.shape[0]:
        raise ValueError(
            f"ego_speed must have 1, {scores.shape[0]}, or an even divisor of "
            f"{scores.shape[0]} values, got {speed.numel()}"
        )

    if stopped_keep_ratio is not None:
        ratio = torch.where(
            speed < stopped_speed,
            ratio.new_full(ratio.shape, float(stopped_keep_ratio)),
            ratio,
        )
    if moderate_keep_ratio is not None:
        ratio = torch.where(
            (speed >= stopped_speed) & (speed < high_speed),
            ratio.new_full(ratio.shape, float(moderate_keep_ratio)),
            ratio,
        )
    if fast_keep_ratio is not None:
        ratio = torch.where(
            speed >= high_speed,
            ratio.new_full(ratio.shape, float(fast_keep_ratio)),
            ratio,
        )
    return ratio.clamp(0.0, 1.0)


def make_budget_mask(
    scores: torch.Tensor,
    default_keep_ratio: float,
    ego_speed: Optional[torch.Tensor] = None,
    stopped_keep_ratio: Optional[float] = None,
    moderate_keep_ratio: Optional[float] = None,
    fast_keep_ratio: Optional[float] = None,
    stopped_speed: float = 0.5,
    high_speed: float = 12.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a top-k keep mask from token scores and optional ego speed."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [B, N].")

    target_ratio = _target_keep_ratio(
        scores,
        default_keep_ratio,
        ego_speed=ego_speed,
        stopped_keep_ratio=stopped_keep_ratio,
        moderate_keep_ratio=moderate_keep_ratio,
        fast_keep_ratio=fast_keep_ratio,
        stopped_speed=stopped_speed,
        high_speed=high_speed,
    )
    num_tokens = scores.shape[1]
    keep_counts = torch.ceil(target_ratio * num_tokens).long().clamp(1, num_tokens)
    sorted_idx = torch.argsort(scores, dim=1, descending=True)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for batch_idx, keep_count in enumerate(keep_counts.tolist()):
        mask[batch_idx, sorted_idx[batch_idx, :keep_count]] = True
    return mask, target_ratio


def apply_dense_budget_mask(
    tokens: torch.Tensor,
    keep_mask: torch.Tensor,
    mode: str = "zero",
) -> Tuple[torch.Tensor, dict]:
    """Apply a keep mask without changing sequence length.

    This is the RepDETR3D-safe form: deformable attention keeps dense spatial
    indexing, while dropped locations are zeroed or filled with ambient context.
    """

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, N, C].")
    if mode not in {"zero", "hard_drop", "ambient_pool"}:
        raise ValueError("mode must be 'zero', 'hard_drop', or 'ambient_pool'.")

    keep_mask = _as_bool_mask(keep_mask, tokens)
    masked = tokens.clone()
    if mode in {"zero", "hard_drop"}:
        masked = masked * keep_mask.unsqueeze(-1).to(dtype=tokens.dtype)
    else:
        ambient = pool_ambient_tokens(tokens, keep_mask, ambient_context_len=1)
        masked = torch.where(keep_mask.unsqueeze(-1), masked, ambient.expand_as(masked))

    keep_count = keep_mask.sum(dim=1)
    metadata = {
        "routing_num_tokens": int(tokens.shape[1]),
        "routing_keep_count": keep_count.detach().cpu().tolist(),
        "routing_keep_ratio_actual": (keep_count.float() / tokens.shape[1])
        .detach()
        .cpu()
        .tolist(),
        "routing_mode": mode,
    }
    return masked, metadata
