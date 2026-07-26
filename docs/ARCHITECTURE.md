 Architecture notes

## High-level design

The implementation should be modular:

```text
Detector
  -> DetectorAdapter
      -> feature extraction hooks
      -> feature injection hooks
  -> TemporalMemoryController
      -> keyframe scheduler
      -> memory cache
      -> PEHandler
      -> MemoryUpdater
      -> optional distillation path
```

The detector itself should remain as unchanged as possible.

## Memory content

Prefer storing appearance features before adding time-dependent 3D positional encoding.

Best case:

```text
appearance_feature_t = backbone_or_neck(images_t)
pe_t = compute_3d_pe(calibration_t, ego_pose_t)
detector_input_t = appearance_feature_t + pe_t
```

Then the memory updater operates on `appearance_feature`, and PE is recomputed for each timestamp.

Fallback if the code only exposes PE-added features:

```text
memory_with_pe_{t-1} -> ego_motion_transform_or_pe_refresh -> updater
```

But this is less clean and should be treated as an ablation, not the default.

## Keyframe schedule

Start with fixed schedules:

- `K=1`: full baseline, should match original detector.
- `K=2`
- `K=3`
- `K=5`

Adaptive keyframe selection should be deferred until fixed schedules work.

## Updater variants

### Variant 0: no-op / full path

Used for debugging.

```text
M_t = FullEncoder(I_t)
```

### Variant 1: naive reuse

Critical baseline.

```text
M_t = warp_or_refresh(M_{t-1})
```

If appearance memory is stored without PE, this may be just:

```text
M_t = M_{t-1}
```

with current-frame PE recomputed downstream.

### Variant 2: gated residual updater

First proposed method.

```text
P_t = ProjectPartial(Z_t_l)
X_t = concat(M_{t-1}, P_t)
G_t = sigmoid(Gate(X_t, ego_embed_t))
R_t = Residual(X_t, ego_embed_t)
M_t = M_{t-1} + G_t * R_t
```

Use lightweight blocks: 1x1 conv/linear, depthwise conv, small MLP, or low-rank adapter.

### Variant 3: cross-attention updater

Second proposed method.

```text
Q = Linear(M_{t-1})
K,V = Linear(P_t)
Delta = CrossAttention(Q, K, V)
M_t = M_{t-1} + Gate * Delta
```

Use this only after the gated residual variant works. Cross-attention can easily become too expensive.

### Variant 4: metadata-conditioned updater

Add ego speed, yaw rate, and time delta as conditioning.

```text
ego_embed = MLP([delta_t, speed, yaw_rate, translation_norm, rotation_angle])
```

Use metadata as conditioning for gates or LayerNorm/FiLM parameters, not as extra dense tokens at first.

Avoid using ground-truth object count/speed for the main method. It can be used only as oracle analysis.



## Learned frame policy

After fixed K schedules are established, the controller may use a FramePolicy to choose the compute path per frame:

- L24: run the full encoder and refresh memory.
- L8, L12, L16: extract the corresponding partial EVA block, project it to the detector feature interface, and update memory.

Implemented policy families:

- fixed_interval: deterministic keyframe interval with one partial action.
- cycle: deterministic sequence such as L24, L8, L12, L16.
- mlp: a small Torch MLP that consumes scalar policy features and predicts difficulty classes. The default class mapping is easy -> L8, medium -> L12, hard -> L24.

MLP policy input variants:

- online metadata only: frame/timestamp/ego-motion features available at inference.
- offline metadata: online metadata plus annotation/object-count features. This is an oracle ablation and not deployable.
- online + L8 cascade: online metadata plus a cheap block-8 similarity probe against cached memory. The L8 probe cost must be included in latency.

For evaluation runs, the MLP policy should load an exported checkpoint. For train-from-scratch or random-policy ablations, set checkpoint=None, load_weights=False, and provide feature_names, hidden_dim, labels, and actions in the config.

## Shape conventions

Support both spatial features and flattened tokens.

Spatial format:

```text
B, N_cam, C, H, W
```

Token format:

```text
B, N_cam, N_token, C
```

Adapters should convert to a canonical format internally and convert back before detector injection.

## Partial ViT feature extraction

The first implementation should extract intermediate features from fixed layers, e.g.:

- 25% depth
- 50% depth
- 75% depth

Do not assume earlier features are detector-ready. They must pass through a projection module and should be trained with feature distillation.

## Practical recommendation

Start with the following minimal method:

```text
appearance memory without PE if possible
fixed K=3
intermediate layer at 50% depth
gated residual updater
teacher feature distillation
RepDETR3D EVA-02 config
```