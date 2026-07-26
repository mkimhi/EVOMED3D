# Experiment plan

## Phase A: repository and baseline sanity

A0. Environment and data

- Confirm the repository runs one validation pass.
- Confirm nuScenes metadata contains ego pose, camera extrinsics/intrinsics, and timestamp.
- Confirm baseline config path for RepDETR3D EVA-02.

A1. Baseline reproduction

- Run baseline inference/validation.
- Record mAP, NDS, latency, GPU memory.
- Profile backbone, neck, head/decoder separately if possible.

A2. Feature interface discovery

- Print/log shapes of full features passed to the detector head.
- Identify whether features are pre-PE or post-PE.
- Identify the layer/block names for EVA-02 intermediate features.

Deliverable: short markdown/code comment with feature shapes and PE location.

## Phase B: memory plumbing

B0. Disabled mode

```text
tm_enabled=false
```

Must match baseline.

B1. Keyframe interval 1

```text
tm_enabled=true
keyframe_interval=1
```

Should match full path.

B2. Naive reuse

```text
tm_enabled=true
keyframe_interval in [2,3,5]
updater=reuse
```

This is the most important baseline for the paper.

## Phase C: partial features

C0. Intermediate feature extraction

Collect partial features from several EVA-02 depths:

```text
25% depth
50% depth
75% depth
```

C1. Partial-only baseline

```text
M_t = ProjectPartial(Z_t_l)
```

No previous memory. This tests whether intermediate features alone are enough.

C2. Feature similarity analysis

For a small validation subset, compute:

- SmoothL1 to full memory
- cosine similarity
- norm ratio
- per-camera similarity
- keyframe vs non-keyframe drift

## Phase D: proposed updater

D0. Gated residual updater

```text
M_t = M_{t-1} + sigmoid(gate) * residual
```

Run with:

```text
K in [2,3,5]
partial_layer in [50%,75%]
lambda_feat in [0.5,1.0]
```

D1. Add ego-motion conditioning

Use relative pose/speed/yaw-rate embedding to condition the gate.

Compare:

```text
without ego_embed
with ego_embed
```

D2. Cross-attention updater

Only after D0 works. Compare accuracy improvement against added latency.

## Phase E: distillation and training schedule

E0. Updater-only training

Freeze backbone. Train updater/projection only.

E1. Updater + neck/head

Freeze backbone. Train updater plus detector neck/head.

E2. Light backbone adaptation

Unfreeze last EVA blocks or add adapters/LoRA if supported.

## Phase F: adaptive schedule, future work

Fixed K is no longer the only adaptive schedule path. The current learned-policy setup predicts frame difficulty and maps it to compute:

1. easy -> L8 partial update
2. medium -> L12 partial update
3. hard -> L24 full encoder refresh

Policy input variants to keep separate in tables:

- online metadata only: frame/timestamp/ego-motion features available at inference.
- offline metadata: online metadata plus annotation/object-count features. Treat this as an oracle ablation.
- online + L8 cascade: online metadata plus a block-8 similarity probe before the final policy decision. Include the probe in latency.

Potential future policy inputs:

- image difference statistics
- previous detection confidence
- previous memory update gate magnitude
- CKA or token-level similarity signals, only when their runtime cost is measured

Avoid relying on ground-truth object count/speed for the main deployable policy.

## Main tables

Table 1: accuracy/latency tradeoff

```text
method | K | partial layer | mAP | NDS | latency | memory | speedup
```

Table 2: ablation

```text
reuse | partial-only | gated | gated+distill | gated+distill+ego | cross-attn
```

Table 3: detector transfer

```text
RepDETR3D EVA-02 | StreamPETR available backbone/config
```

## Stop conditions

Stop or redesign if:

1. Naive reuse is as good as updater under equal latency.
2. Updater latency erases the encoder savings.
3. Distillation similarity improves but detection does not.
4. PE handling cannot be made stable.

If this happens, pivot to adaptive keyframe scheduling or token-level residual refinement rather than dense memory update.
