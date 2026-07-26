# Detector adapter design: StreamPETR and RepDETR3D

## Motivation

The project should not be tied to one detector. StreamPETR is the conceptual baseline, but RepDETR3D is a practical first target because the available repository contains EVA-02 RepDETR3D configurations.

The temporal memory logic should sit at the feature interface:

```text
image backbone / neck features <-> detector head / decoder
```

## Adapter interface

Define a detector adapter with methods like:

```python
class DetectorAdapter:
    def extract_full_features(self, model, batch):
        """Run the original full feature extraction path."""

    def extract_partial_features(self, model, batch, layer_id):
        """Run partial backbone or collect intermediate features."""

    def compute_position_encoding(self, model, batch, feature_spec):
        """Compute timestamp-current 3D PE if the detector uses it."""

    def inject_features(self, model, batch, features, pe=None):
        """Run the detector head/decoder using supplied features."""

    def feature_to_memory(self, features):
        """Convert detector features to canonical memory format."""

    def memory_to_feature(self, memory, feature_spec):
        """Convert canonical memory back to detector format."""
```

The exact API can follow the repository style, but keep these responsibilities separated.

## Canonical memory spec

Create a small data object or dictionary:

```python
MemoryPacket = {
    "features": ...,          # tensor or list of tensors
    "format": "spatial" or "tokens",
    "pre_pe": True/False,
    "timestamp": ...,
    "ego_pose": ...,
    "cam_params": ...,
    "shape_meta": ...,
}
```

## RepDETR3D first

Initial implementation target:

```text
projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e.py
```

Tasks:

1. Locate EVA-02 backbone feature extraction.
2. Locate the feature format passed to the RepDETR3D head.
3. Add hooks to collect intermediate EVA block outputs.
4. Add a config flag to choose partial layer.
5. Add memory updater before the detector head consumes features.
6. Preserve the original config as baseline.

## StreamPETR later

For StreamPETR:

1. Locate where image features are extracted.
2. Locate where 3D PE is computed and added.
3. Prefer caching pre-PE appearance features.
4. Keep StreamPETR's object-query temporal memory unchanged.
5. Add dense image memory update as an additional path.

## Important compatibility rule

Do not change detector semantics when memory is disabled.

These modes must exist:

```text
tm_enabled=false              # exact baseline
tm_enabled=true, K=1          # should match full path
tm_enabled=true, updater=reuse
tm_enabled=true, updater=gated_residual
tm_enabled=true, updater=cross_attention
```

## Multi-scale features

If the detector uses multi-scale features, memory must support a list of tensors. Start with the highest-impact scale if needed, but document this limitation.

Potential variants:

1. Update all scales.
2. Update only final/highest-resolution scale.
3. Update final scale and recompute cheap FPN layers.

## Pretrained weights

Do not assume StreamPETR EVA-02 weights are available. The project should be able to:

1. Load available RepDETR3D EVA-02 weights/configs.
2. Load original StreamPETR weights for non-EVA baselines.
3. Train only new modules plus neck/head when pretrained weights are incomplete.
4. Optionally initialize the updater using teacher distillation from a full RepDETR3D/StreamPETR run.