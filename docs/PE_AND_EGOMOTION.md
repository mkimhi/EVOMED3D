# 3D positional encoding and ego-motion notes

## Why this matters

PETR/StreamPETR-style detectors rely heavily on 3D position embeddings derived from camera geometry and 3D coordinates. A cached memory from time `t-1` may be geometrically stale at time `t` because the ego vehicle and camera coordinate relationship changed.

If cached features already include 3D PE, directly feeding `Memory_{t-1}` into the updater can mix appearance and stale geometry.

## Preferred design: separate appearance and PE

The preferred implementation is:

```text
A_t = appearance features from image backbone/neck
PE_t = 3D position embedding computed for timestamp t
D_t = detector input = A_t + PE_t
```

Cache and update `A_t`, not `D_t`.

For non-keyframes:

```text
A_t_hat = MemoryUpdater(A_{t-1}, partial_feature_t, ego_embed_t)
PE_t = compute_3d_pe(timestamp=t)
D_t = A_t_hat + PE_t
```

This avoids explicit transformation of PE-added memory.

## If only PE-added features are available

If the code path only exposes features after PE addition, implement a conservative PE handler:

1. Try to locate where PE is computed and refactor so memory is cached before PE.
2. If refactor is too invasive, add a temporary mode called `pe_mode=stale_pe` for debugging only.
3. Add `pe_mode=recompute_pe` as soon as possible.
4. Add `pe_mode=ego_transform_pe` only if recomputing PE is impossible or as an ablation.

Do not make stale PE the main method.

## Ego-motion transform

Let `T_global_ego_t` be the ego pose at time `t` and `T_ego_global_t = inverse(T_global_ego_t)`.

A point represented in ego coordinates at `t-1` can be transformed into ego coordinates at `t` by:

```text
p_ego_t = T_ego_global_t @ T_global_ego_{t-1} @ p_ego_{t-1}
```

The relative transform is:

```text
T_t_from_tminus1 = inverse(T_global_ego_t) @ T_global_ego_{t-1}
```

Use the dataset's exact pose convention. Add tests before trusting this.

## What should be transformed?

Options:

### Option A: recompute PE at current time

Recommended.

```text
A_{t-1} is appearance memory
PE_t is recomputed from current calibration/ego pose
```

No explicit memory warp is needed unless experiments show severe motion sensitivity.

### Option B: condition updater on ego-motion

Recommended early.

```text
ego_embed = MLP(relative_translation, relative_rotation, delta_t, speed, yaw_rate)
```

Use this embedding to modulate the update gate.

### Option C: spatially warp memory

Potentially useful, but more complex. For spatial image features, warping memory across views is not trivial because 3D scene geometry/depth is uncertain. Avoid as the first implementation.

### Option D: transform PE-added memory

Least preferred. This assumes that PE and appearance are separable after addition, which is not generally true after nonlinear layers.

## Tests

Required tests:

1. Identity ego-motion should leave transformed coordinates unchanged.
2. Pure translation should move coordinates according to the expected sign convention.
3. Pure yaw rotation should match a hand-computed 2D rotation in the ground plane.
4. `keyframe_interval=1` should match baseline outputs.
5. `pe_mode=separate_recompute` should not change baseline behavior when memory is disabled.

## Implementation instruction

Before coding the updater, search the detector code for where 3D PE is created and added. Create a small note in the PR/commit message describing whether the memory is pre-PE or post-PE.