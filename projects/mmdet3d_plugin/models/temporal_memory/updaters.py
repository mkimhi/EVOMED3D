import torch
import torch.nn.functional as F
from torch import nn
from .policies import action_to_layer, normalize_policy_layer


class ReuseUpdater:
    def __call__(self, previous_memory, *args, **kwargs):
        return previous_memory


class PartialOnlyUpdater:
    def __call__(self, previous_memory, partial_current=None, **kwargs):
        if partial_current is None:
            raise ValueError('partial_current is required for partial-only updates')
        return partial_current


class GatedResidualMemoryUpdater(nn.Module):
    def __init__(
        self,
        channels,
        ego_dim=0,
        feature_format='spatial',
        residual_scale_init=0.0,
    ):
        super().__init__()
        if feature_format not in ('spatial', 'tokens'):
            raise ValueError("feature_format must be 'spatial' or 'tokens'")
        self.channels = channels
        self.ego_dim = ego_dim
        self.feature_format = feature_format
        in_channels = channels * 2 + ego_dim
        if feature_format == 'spatial':
            self.gate = nn.Conv2d(in_channels, channels, kernel_size=1)
            self.residual = nn.Conv2d(in_channels, channels, kernel_size=1)
        else:
            self.gate = nn.Linear(in_channels, channels)
            self.residual = nn.Linear(in_channels, channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, previous_memory, partial_current=None, ego_motion_embedding=None):
        if isinstance(previous_memory, list):
            return [
                self.forward(prev, part, ego_motion_embedding)
                for prev, part in zip(previous_memory, partial_current)
            ]
        if isinstance(previous_memory, tuple):
            return tuple(
                self.forward(prev, part, ego_motion_embedding)
                for prev, part in zip(previous_memory, partial_current)
            )
        if partial_current is None:
            raise ValueError('partial_current is required for gated residual updates')
        if previous_memory.shape != partial_current.shape:
            raise ValueError(
                f'partial_current shape {tuple(partial_current.shape)} must match '
                f'previous_memory shape {tuple(previous_memory.shape)}')

        x = self._concat_inputs(previous_memory, partial_current, ego_motion_embedding)
        gate = torch.sigmoid(self.gate(x))
        residual = self.residual(x)
        if self.feature_format == 'spatial':
            gate = self._from_spatial(gate, previous_memory)
            residual = self._from_spatial(residual, previous_memory)
        return previous_memory + self.residual_scale * gate * residual

    def _concat_inputs(self, previous_memory, partial_current, ego_motion_embedding):
        if self.feature_format == 'spatial':
            if previous_memory.dim() != 5:
                raise ValueError('spatial memory must have shape B,N_cam,C,H,W')
            pieces = [previous_memory, partial_current]
            if ego_motion_embedding is not None:
                pieces.append(self._expand_spatial_ego(ego_motion_embedding, previous_memory))
            return self._to_spatial(torch.cat(pieces, dim=2))

        if previous_memory.dim() != 4:
            raise ValueError('token memory must have shape B,N_cam,N_token,C')
        pieces = [previous_memory, partial_current]
        if ego_motion_embedding is not None:
            pieces.append(self._expand_token_ego(ego_motion_embedding, previous_memory))
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def _to_spatial(x):
        b, n_cam, channels, h, w = x.shape
        return x.reshape(b * n_cam, channels, h, w)

    @staticmethod
    def _from_spatial(x, reference):
        b, n_cam, _, h, w = reference.shape
        return x.reshape(b, n_cam, -1, h, w)

    @staticmethod
    def _expand_spatial_ego(ego_motion_embedding, reference):
        b, n_cam, _, h, w = reference.shape
        return ego_motion_embedding[:, None, :, None, None].expand(b, n_cam, -1, h, w)

    @staticmethod
    def _expand_token_ego(ego_motion_embedding, reference):
        b, n_cam, n_token, _ = reference.shape
        return ego_motion_embedding[:, None, None, :].expand(b, n_cam, n_token, -1)


class LayeredMemoryUpdater(nn.Module):
    def __init__(self, updater_cls, layers=(8, 12, 16), shared_memory_gate=True, **updater_cfg):
        super().__init__()
        self.layers = [normalize_policy_layer(layer) for layer in layers]
        self.shared_memory_gate = shared_memory_gate
        if shared_memory_gate:
            self.shared_updater = updater_cls(**updater_cfg)
            self.updaters = {layer: self.shared_updater for layer in self.layers}
        else:
            self.updaters = nn.ModuleDict({
                layer: updater_cls(**updater_cfg)
                for layer in self.layers
            })

    def forward(self, previous_memory, partial_current=None, action=None, **kwargs):
        layer = action_to_layer(action) if action is not None else self.layers[0]
        if layer not in self.updaters:
            raise KeyError(f'Unsupported temporal memory layer action: {action}')
        return self.updaters[layer](
            previous_memory,
            partial_current=partial_current,
            **kwargs)


class AttentionMemoryUpdater(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        kv_pool_size=(8, 20),
        residual_scale_init=0.0,
        use_gate=False,
        per_camera=False,
        inverted=False,
    ):
        super().__init__()
        self.channels = channels
        self.kv_pool_size = tuple(kv_pool_size)
        self.use_gate = use_gate
        self.per_camera = per_camera
        self.inverted = inverted
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True)
        self.out_proj = nn.Linear(channels, channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        if use_gate:
            self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        else:
            self.gate = None

    def forward(self, previous_memory, partial_current=None, ego_motion_embedding=None):
        if isinstance(previous_memory, list):
            return [
                self.forward(prev, part, ego_motion_embedding)
                for prev, part in zip(previous_memory, partial_current)
            ]
        if isinstance(previous_memory, tuple):
            return tuple(
                self.forward(prev, part, ego_motion_embedding)
                for prev, part in zip(previous_memory, partial_current)
            )
        if partial_current is None:
            raise ValueError('partial_current is required for attention updates')
        if previous_memory.shape != partial_current.shape:
            raise ValueError(
                f'partial_current shape {tuple(partial_current.shape)} must match '
                f'previous_memory shape {tuple(previous_memory.shape)}')
        if previous_memory.dim() != 5:
            raise ValueError('attention memory must have shape B,N_cam,C,H,W')

        if self.per_camera:
            attn_out = self._attend_per_camera(previous_memory, partial_current)
        else:
            attn_out = self._attend_all_cameras(previous_memory, partial_current)
        delta = self._tokens_to_memory(self.out_proj(attn_out), previous_memory)
        if self.gate is not None:
            delta = delta * self._gate(previous_memory, partial_current)
        return previous_memory + self.residual_scale * delta

    def _attend_all_cameras(self, previous_memory, partial_current):
        q_source = partial_current if self.inverted else previous_memory
        kv_source = previous_memory if self.inverted else partial_current
        q = self._memory_to_tokens(q_source)
        kv = self._pooled_memory_to_tokens(kv_source)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out

    def _attend_per_camera(self, previous_memory, partial_current):
        q_source = partial_current if self.inverted else previous_memory
        kv_source = previous_memory if self.inverted else partial_current
        q = self._memory_to_camera_tokens(q_source)
        kv = self._pooled_memory_to_camera_tokens(kv_source)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self._camera_tokens_to_tokens(out, previous_memory)

    @staticmethod
    def _memory_to_tokens(x):
        b, n_cam, c, h, w = x.shape
        return x.permute(0, 1, 3, 4, 2).reshape(b, n_cam * h * w, c)

    @staticmethod
    def _tokens_to_memory(tokens, reference):
        b, n_cam, c, h, w = reference.shape
        return tokens.reshape(b, n_cam, h, w, c).permute(0, 1, 4, 2, 3)

    def _pooled_memory_to_tokens(self, x):
        b, n_cam, c, h, w = x.shape
        flat = x.reshape(b * n_cam, c, h, w)
        pooled = F.adaptive_avg_pool2d(flat, self.kv_pool_size)
        ph, pw = pooled.shape[-2:]
        return pooled.reshape(b, n_cam, c, ph, pw).permute(0, 1, 3, 4, 2).reshape(
            b, n_cam * ph * pw, c)

    @staticmethod
    def _memory_to_camera_tokens(x):
        b, n_cam, c, h, w = x.shape
        return x.permute(0, 1, 3, 4, 2).reshape(b * n_cam, h * w, c)

    def _pooled_memory_to_camera_tokens(self, x):
        b, n_cam, c, h, w = x.shape
        flat = x.reshape(b * n_cam, c, h, w)
        pooled = F.adaptive_avg_pool2d(flat, self.kv_pool_size)
        ph, pw = pooled.shape[-2:]
        return pooled.permute(0, 2, 3, 1).reshape(b * n_cam, ph * pw, c)

    @staticmethod
    def _camera_tokens_to_tokens(tokens, reference):
        b, n_cam, _, h, w = reference.shape
        c = tokens.size(-1)
        return tokens.reshape(b, n_cam, h * w, c).reshape(b, n_cam * h * w, c)

    def _gate(self, previous_memory, partial_current):
        b, n_cam, _, h, w = previous_memory.shape
        x = torch.cat([previous_memory, partial_current], dim=2).reshape(
            b * n_cam, self.channels * 2, h, w)
        return torch.sigmoid(self.gate(x)).reshape(b, n_cam, self.channels, h, w)


def build_memory_updater(spec, **kwargs):
    if callable(spec) and not isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        spec = spec.copy()
        name = spec.pop('type', spec.pop('name', 'reuse'))
        spec.update(kwargs)
    else:
        name = spec
        spec = kwargs

    if name == 'reuse':
        return ReuseUpdater()
    if name == 'partial_only':
        return PartialOnlyUpdater()
    if name == 'gated_residual':
        if 'layers' in spec or 'shared_memory_gate' in spec:
            layers = spec.pop('layers', (8, 12, 16))
            shared_memory_gate = spec.pop('shared_memory_gate', True)
            return LayeredMemoryUpdater(
                GatedResidualMemoryUpdater,
                layers=layers,
                shared_memory_gate=shared_memory_gate,
                **spec)
        return GatedResidualMemoryUpdater(**spec)
    if name == 'attention_residual':
        return AttentionMemoryUpdater(**spec)
    if name == 'gated_attention_residual':
        return AttentionMemoryUpdater(use_gate=True, **spec)
    if name == 'per_camera_attention_residual':
        return AttentionMemoryUpdater(per_camera=True, **spec)
    if name == 'inverted_attention_residual':
        return AttentionMemoryUpdater(inverted=True, **spec)
    raise KeyError(f'Unsupported temporal memory updater: {name}')
