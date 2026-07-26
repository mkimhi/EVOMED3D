from ..detector_adapters import DetectorAdapter
from ..memory_packet import MemoryPacket
from ..partial_features import resolve_eva_layer_index


class RepDetr3DAdapter(DetectorAdapter):
    memory_format = 'spatial'
    memory_is_pre_pe = True
    pe_mode = 'separate_recompute'
    pe_source = 'head_or_transformer'

    def extract_full_features(self, model, batch, len_queue=1, training_mode=False):
        return model.extract_feat(batch['img'], len_queue, training_mode)

    def extract_partial_features(self, model, batch, layer_id, depth=24, **kwargs):
        original_img = batch['img']
        img = self._flatten_img(original_img)
        if getattr(model, 'use_grid_mask', False):
            img = model.grid_mask(img)
        layer_index = resolve_eva_layer_index(layer_id, depth=depth)
        feature = model.img_backbone.forward_intermediate(img, layer_index)
        self._assert_partial_feature(feature)
        return self._reshape_partial_feature(feature, original_img)

    def inject_features(self, model, batch, features, img_metas=None):
        self._assert_head_features(features)
        data = dict(batch)
        data['img_feats'] = features
        if img_metas is None:
            img_metas = data.pop('img_metas', None)
        assert img_metas is not None, 'RepDETR3D feature injection requires img_metas.'
        return model.pts_bbox_head(img_metas, **data)

    def feature_to_memory(self, features, **kwargs):
        self._assert_head_features(features)
        meta = dict(kwargs.pop('meta', {}))
        meta.setdefault('pe_mode', self.pe_mode)
        meta.setdefault('pe_source', self.pe_source)
        return MemoryPacket.from_features(
            features,
            format=self.memory_format,
            pre_pe=self.memory_is_pre_pe,
            meta=meta,
            **kwargs,
        )

    def _assert_head_features(self, features):
        if not self.debug_shapes:
            return
        assert isinstance(features, (list, tuple)), 'RepDETR3D features must be a list of tensors.'
        for index, feature in enumerate(features):
            assert feature.dim() == 5, (
                f'RepDETR3D feature level {index} must be 5D [B, N, C, H, W]; '
                f'got {tuple(feature.shape)}.'
            )

    def _assert_partial_feature(self, feature):
        if not self.debug_shapes:
            return
        assert feature.dim() == 4, (
            f'RepDETR3D partial EVA feature must be 4D [B*N, C, H, W]; '
            f'got {tuple(feature.shape)}.'
        )

    def _flatten_img(self, img):
        if img.dim() == 6:
            img = img.flatten(1, 2)
        if img.dim() == 5 and img.size(0) == 1:
            img = img.squeeze(0)
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        return img

    def _reshape_partial_feature(self, feature, original_img):
        B = original_img.size(0)
        C, H, W = feature.shape[1:]
        if original_img.dim() == 6:
            T, N = original_img.size(1), original_img.size(2)
            return feature.view(B, T, N, C, H, W)
        if original_img.dim() == 5:
            N = int(feature.size(0) / B)
            return feature.view(B, N, C, H, W)
        return feature
