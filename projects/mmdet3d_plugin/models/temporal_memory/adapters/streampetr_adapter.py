from ..detector_adapters import DetectorAdapter
from ..partial_features import resolve_eva_layer_index


class StreamPETRAdapter(DetectorAdapter):
    """Adapter for StreamPETR's dense, pre-position-encoding image feature.

    TODO: Implement PE recomputation around
    projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py::StreamPETRHead.position_embeding
    and keep cached dense memory pre-PE.

    TODO: Implement feature injection through
    projects/mmdet3d_plugin/models/detectors/petr3d.py::Petr3D.prepare_location,
    projects/mmdet3d_plugin/models/detectors/petr3d.py::Petr3D.forward_roi_head, and
    projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py::StreamPETRHead.forward
    without changing StreamPETR object-query temporal memory semantics.
    """

    memory_format = 'spatial'
    memory_is_pre_pe = True
    pe_mode = 'separate_recompute'
    pe_source = 'streampetr_head_position_embeding'

    _PE_TODO = (
        'StreamPETR PE recomputation is not wired yet. TODO: refactor or wrap '
        'projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py::'
        'StreamPETRHead.position_embeding so dense memory stays pre-PE.'
    )
    _INJECT_TODO = (
        'StreamPETR feature injection is not wired yet. TODO: route supplied '
        'features through projects/mmdet3d_plugin/models/detectors/petr3d.py::'
        'Petr3D.prepare_location, Petr3D.forward_roi_head, and '
        'projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py::'
        'StreamPETRHead.forward.'
    )

    def extract_full_features(self, model, batch, **kwargs):
        features = model.extract_img_feat(
            batch['img'], len_queue=1, training_mode=model.training)
        if hasattr(features, 'dim') and features.dim() == 6 and features.size(1) == 1:
            features = features[:, 0]
        return self._validate_spatial_feature(features)

    def extract_partial_features(self, model, batch, layer_id, depth=24, **kwargs):
        layer_index = resolve_eva_layer_index(layer_id, depth=depth)
        features = model.extract_img_feat_intermediate(
            batch['img'], layer_index, len_queue=1)
        return self._validate_spatial_feature(features)

    def extract_projected_partial_features(
            self, model, batch, layer_id, depth=24, **kwargs):
        """Return an intermediate feature at the detector's pre-PE boundary.

        ``Petr3D.extract_img_feat_intermediate`` applies EVA's existing
        ``sim_fpn`` projection, so no second projector should be applied here.
        """
        return self.extract_partial_features(
            model, batch, layer_id, depth=depth, **kwargs)

    def compute_position_encoding(self, model, batch, feature_spec):
        raise NotImplementedError(self._PE_TODO)

    def inject_features(self, model, batch, features, **kwargs):
        raise NotImplementedError(self._INJECT_TODO)

    def feature_to_memory(self, features, **kwargs):
        return self._validate_spatial_feature(features)

    def memory_to_feature(self, memory, feature_spec=None):
        return self._validate_spatial_feature(memory)

    @staticmethod
    def _validate_spatial_feature(features):
        if not hasattr(features, 'dim') or features.dim() != 5:
            shape = getattr(features, 'shape', None)
            raise ValueError(
                'StreamPETR expected [B, N_cam, C, H, W] spatial features; '
                f'got {shape}.')
        return features
