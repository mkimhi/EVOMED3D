from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class MemoryPacket:
    features: Any
    format: str
    pre_pe: bool
    timestamp: Any = None
    ego_pose: Any = None
    cam_params: Any = None
    shape_meta: Optional[Sequence[Tuple[int, ...]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_features(cls, features, *, format, pre_pe, **kwargs):
        if isinstance(features, (list, tuple)):
            shape_meta: List[Tuple[int, ...]] = [tuple(feature.shape) for feature in features]
        else:
            shape_meta = [tuple(features.shape)]
        return cls(features=features, format=format, pre_pe=pre_pe, shape_meta=shape_meta, **kwargs)
