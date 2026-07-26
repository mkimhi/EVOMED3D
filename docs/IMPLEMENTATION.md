# Temporal Memory Implementation Plan

This plan tracks the next RepDETR3D EVA-02 temporal-memory implementation phase.

## Current runnable state

- Full RepDETR3D EVA-02 baseline and naive reuse experiments are runnable.
- Inference routing for non-reuse temporal modes now extracts partial EVA features, projects them to the detector feature interface, and passes them into the temporal controller.
- Training routing for non-reuse modes now stacks per-frame temporal features and adds `loss_tm_distill` when the controller produces teacher distillation loss.
- Initial configs exist for partial-only, gated residual, deterministic cycle, and MLP policy variants. The MLP policy supports online metadata, offline metadata oracle, and online + L8 cascade inputs.
- `RepDETR3D EVA02 gated residual K=2 + distill`, partial-only, projection-only, ego-conditioned, and cross-attention rows in `RESULTS.md` should not be treated as trained results until their checkpoints are produced.

## Immediate test-only command after implementation

After a gated residual K=2 checkpoint exists, run:

```bash
bash tools/dist_test.sh projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k2_l8_distill.py <TRAINED_GATED_K2_CKPT> 2 --eval bbox --cfg-options model.tm_debug_latency=True
```

Keep `--cfg-options` as the final Python argument.

## Implementation phases

### 1. Feature interface audit

- [x] Confirm full RepDETR3D detector feature shapes passed to `SparseHead`.
- [x] Confirm partial EVA block outputs for block `8`.
- [ ] Confirm partial EVA block outputs for blocks `12` and `16`.
- [x] Add tests for full feature, partial feature, and projected feature shape compatibility.

### 2. Partial feature projection

- [x] Add a trainable `PartialFeatureProjector`.
- [x] Support configurable `partial_layer`, including `block8` and integer block IDs.
- [ ] Add concrete configs/tests for `block12` and `block16`.
- [x] Project raw EVA intermediate features to the RepDETR3D detector feature interface: a multi-scale list of `[B, N_cam, 256, H, W]` tensors.
- [x] Use block 8 as the first cheap default.
- [ ] Add block 12 and block 16 ablations.

### 3. Partial-only mode

- [x] Add `updater='partial_only'`.
- On non-keyframes:

```text
M_t = ProjectPartial(EVA_block_l(I_t))
```

- [x] Do not use previous memory. This tests whether current partial features alone can replace full features between keyframes.
- [x] Add `repdetr3d_eva02_800_bs2_seq_24e_tm_partial_only_l8.py`.

### 4. Training and inference routing

- [x] Update RepDETR3D inference routing so non-reuse updaters actually use partial extraction and updater logic.
- [x] Update RepDETR3D training routing so non-reuse updaters actually use partial extraction and updater logic.
- [x] Keep keyframes on the full encoder.
- [x] Keep `keyframe_interval=1` matching the full path.
- [x] Preserve baseline behavior when temporal memory is disabled.

### 5. Distillation

- [x] During training on non-keyframes, optionally run the full encoder teacher under `torch.no_grad()`.
- [x] Distill student projected/updated features to stop-gradient full features with SmoothL1 at the detector feature interface.
- [x] Add `loss_tm_distill` into RepDETR3D training losses.
- [x] Keep the inference assertion that the full teacher is not called on non-keyframes.

### 6. Gated residual updater

- Use:

```text
M_t = M_{t-1} + alpha * sigmoid(gate([M_{t-1}, P_t])) * residual([M_{t-1}, P_t])
```

- [x] Keep conservative initialization (`residual_scale_init=0.0` or small).
- [x] Add `repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k2_l8_distill.py`.
- [ ] Add K=3 and K=5 block-8 configs.
- [ ] Train K=2, K=3, and K=5, starting with block 8.

### 7. Cross-attention updater

- [ ] Add a replaceable updater:

```text
Q = memory
K,V = projected partial features
M_t = M_{t-1} + gated_cross_attention(Q, K, V)
```

- [ ] Run only after the gated residual version has a trained baseline because cross-attention may erase latency gains.

### 8. Configs

Add new configs instead of mutating baseline configs:

- [ ] projection-only L8/L12/L16
- [x] partial-only L8
- [ ] partial-only L12/L16
- [x] gated residual K2 L8 + distill
- [ ] gated residual K3/K5 L8 + distill
- [ ] gated residual K3 L12/L16 + distill
- [ ] gated residual K3 L8 + distill + ego
- [ ] cross-attention K3 L8 + distill

All training configs should start from:

```python
load_from = './ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth'
```


### 8a. Learned frame policy

- [x] Add MlpFramePolicy behind TemporalMemoryController.
- [x] Pass scalar policy_features from RepDETR3D routing into the controller.
- [x] Support policy checkpoints exported by tools/train_temporal_policy_mlp.py.
- [x] Support checkpoint=None with load_weights=False for train-from-scratch or random-policy ablations.
- [x] Add no-distillation configs for online metadata, offline metadata, and online + L8 cascade policies.
- [ ] Train and evaluate the three learned-policy detector configs, including latency with any L8 probe cost.

### 9. Minimum tests before training

- [x] `keyframe_interval=1` matches the full path.
- [x] Partial projector output shape equals detector feature shape.
- [x] Non-reuse updater is called on non-keyframes.
- [x] Teacher path is stop-gradient.
- [x] Inference teacher assertion fires if teacher is called.
- [ ] A real short train/test sequence runs without crashing.
