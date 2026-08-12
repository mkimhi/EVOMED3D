# Predictive Tri-State Token Routing Roadmap

This is the fixed-token routing track migrated from SEPatch3D. It excludes the
older adaptive patch-size direction.

## Objective

Reduce the ViT token-processing bottleneck in multi-view 3D detection while
preserving:

- dynamic hazard visibility;
- global ambient context such as weather, lighting, and scene priors;
- temporal stability of routing masks;
- compatibility with multi-view feature geometry.

Patch size stays fixed, initially at 16x16. Compute savings should come from
token routing and sequence-length reduction, not variable patch tokenization.

## Router States

### Foveate / Keep

Tokens selected by top-down intent. Later versions can use future waypoints,
ego velocity, yaw, or planned trajectory.

### Bypass / Safety Override

Tokens preserved by bottom-up temporal variance or anomaly detection. This is
intended to keep peripheral hazards that pure top-down routing might miss.

### Merge / Ambient Context

Tokens not selected by keep or bypass are pooled into ambient context instead
of being discarded.

## Core Modules

1. Intent encoder: a lightweight module that maps kinematic intent to routing
   logits.
2. Anomaly filter: a temporal-difference path, ideally after ego-motion
   compensation.
3. Differentiable router: hard Gumbel routing for trainable discrete masks.
4. Ambient pooler: average-pool dropped tokens into one or more context tokens.
5. Budget loss: penalize keep ratio above the configured system budget.

## Initial Ablations

1. Fixed-token baseline with routing disabled.
2. Random or score-based hard drop at the same token budget.
3. Ambient pooling at the same token budget.
4. Speed-conditioned budgets.
5. Trainable router with a budget penalty.

## Non-Goals

- Do not implement adaptive patch-size routing.
- Do not make DINO-style distillation required for baseline runs.
- Do not compare patch-size changes as part of the first routing ablations.
