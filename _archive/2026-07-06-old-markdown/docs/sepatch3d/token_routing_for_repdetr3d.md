# Token Routing For RepDETR3D

This migration adds the reusable SEPatch3D tri-state routing utilities to
StreamPETR and exposes an opt-in RepDETR3D config:

```text
projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_router.py
```

## What Is Implemented

- Fixed patch size remains unchanged.
- `SparseHead` can compute token scores from flattened multi-level image
  features.
- Speed-conditioned keep ratios can use `ego_speed` from the nuScenes dataset.
- `routing_mode='ambient_pool'` fills dropped dense locations with pooled
  ambient context.
- `routing_mode='zero'` and `routing_mode='hard_drop'` zero dropped dense
  locations while preserving tensor shape.

## Important Limitation

RepDETR3D uses deformable CUDA attention with dense `spatial_flatten` and
`level_start_index` indexing. Arbitrarily shortening `feat_flatten` would break
the mapping from projected sample points to feature-map locations.

For that reason, the integrated hook currently preserves dense feature geometry.
It is useful for validating routing masks, budget schedules, and ambient
context behavior, but it does not yet reduce the CUDA attention sequence length.

True sequence-length reduction needs a sparse-aware attention path that carries
token positions through the projection/sampling step.

## Config Knobs

The router config sets these under `model.pts_bbox_head`:

```python
routing_mode='ambient_pool'
routing_keep_ratio=0.5
routing_keep_ratio_stopped=0.5
routing_keep_ratio_moderate=0.6
routing_keep_ratio_fast=0.7
routing_stopped_speed=0.5
routing_high_speed=12.0
```

Use the original RepDETR3D config for unchanged baseline behavior.
