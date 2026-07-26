import torch
from torch import nn

from projects.mmdet3d_plugin.models.backbones.eva_vit import SimpleFeaturePyramid
from .policies import action_to_layer, normalize_policy_layer


class PartialFeatureProjector(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=256,
        scale_factors=(4, 2, 1, 0.5),
        out_indices=(2, 3, 4, 5),
        norm="LN",
    ):
        super().__init__()
        self.pyramid = SimpleFeaturePyramid(
            scale_factors=list(scale_factors),
            in_channels=in_channels,
            out_channels=out_channels,
            out_indices=list(out_indices),
            norm=norm,
        )

    def forward(self, partial_features):
        if partial_features.dim() == 5:
            b, n_cam, c, h, w = partial_features.shape
            flat = partial_features.reshape(b * n_cam, c, h, w)
            return [
                feature.reshape(b, n_cam, feature.size(1), feature.size(2), feature.size(3))
                for feature in self.pyramid(flat)
            ]
        if partial_features.dim() == 6:
            b, t, n_cam, c, h, w = partial_features.shape
            flat = partial_features.reshape(b * t * n_cam, c, h, w)
            return [
                feature.reshape(b, t, n_cam, feature.size(1), feature.size(2), feature.size(3))
                for feature in self.pyramid(flat)
            ]
        raise ValueError('partial_features must have shape B,N_cam,C,H,W or B,T,N_cam,C,H,W')


class LayeredPartialFeatureProjector(nn.Module):
    def __init__(self, layers=(8, 12, 16), projector_cfg=None, per_layer_cfg=None):
        super().__init__()
        projector_cfg = dict(projector_cfg or {})
        per_layer_cfg = dict(per_layer_cfg or {})
        self.projectors = nn.ModuleDict()
        for layer in layers:
            key = normalize_policy_layer(layer)
            cfg = dict(projector_cfg)
            cfg.update(per_layer_cfg.get(key, per_layer_cfg.get(str(layer), {})))
            self.projectors[key] = PartialFeatureProjector(**cfg)

    def forward(self, partial_features, layer_id=None, action=None):
        layer = action_to_layer(action) if action is not None else normalize_policy_layer(layer_id)
        if layer is None:
            raise ValueError('layer_id or action is required for layered partial projection')
        if layer not in self.projectors:
            raise KeyError(f'Unsupported partial feature layer: {layer}')
        return self.projectors[layer](partial_features)
