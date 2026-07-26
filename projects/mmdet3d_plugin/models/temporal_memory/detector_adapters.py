from abc import ABC, abstractmethod


class DetectorAdapter(ABC):
    def __init__(self, debug_shapes=False):
        self.debug_shapes = debug_shapes

    @abstractmethod
    def extract_full_features(self, model, batch, **kwargs):
        """Run the detector's existing full feature extraction path."""

    def extract_partial_features(self, model, batch, layer_id, **kwargs):
        raise NotImplementedError('partial feature extraction is not implemented yet')

    def compute_position_encoding(self, model, batch, feature_spec):
        return None

    @abstractmethod
    def inject_features(self, model, batch, features, **kwargs):
        """Run the detector head using supplied features."""

    @abstractmethod
    def feature_to_memory(self, features, **kwargs):
        """Convert detector features to a canonical memory packet."""

    def memory_to_feature(self, memory, feature_spec=None):
        return memory.features


def build_detector_adapter(name, **kwargs):
    if name == 'repdetr3d':
        from .adapters.repdetr3d_adapter import RepDetr3DAdapter
        return RepDetr3DAdapter(**kwargs)
    if name == 'streampetr':
        from .adapters.streampetr_adapter import StreamPETRAdapter
        return StreamPETRAdapter(**kwargs)
    raise KeyError(f'Unsupported temporal memory detector adapter: {name}')
