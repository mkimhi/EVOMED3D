from ..detector_adapters import DetectorAdapter


class StreamPETRAdapter(DetectorAdapter):
    """StreamPETR adapter placeholder for dense temporal memory.

    TODO: Implement feature extraction in
    projects/mmdet3d_plugin/models/detectors/petr3d.py::Petr3D.extract_img_feat
    once the StreamPETR dense feature level is confirmed for each supported
    backbone/config.

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

    _FEATURE_TODO = (
        'StreamPETR dense feature extraction is not wired yet. TODO: verify '
        'projects/mmdet3d_plugin/models/detectors/petr3d.py::Petr3D.extract_img_feat '
        'and the config-specific position_level before enabling this adapter.'
    )
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
        raise NotImplementedError(self._FEATURE_TODO)

    def extract_partial_features(self, model, batch, layer_id, **kwargs):
        raise NotImplementedError(self._FEATURE_TODO)

    def compute_position_encoding(self, model, batch, feature_spec):
        raise NotImplementedError(self._PE_TODO)

    def inject_features(self, model, batch, features, **kwargs):
        raise NotImplementedError(self._INJECT_TODO)

    def feature_to_memory(self, features, **kwargs):
        raise NotImplementedError(self._FEATURE_TODO)

    def memory_to_feature(self, memory, feature_spec=None):
        raise NotImplementedError(self._INJECT_TODO)
