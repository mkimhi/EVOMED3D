# Temporal-Memory Depth and Token Routers Design

## Goal

Add two detector-agnostic temporal-memory routing alternatives for RepDETR3D EVA-02 without changing the full-path baseline:

1. A sample-level, four-exit depth router that chooses exactly one of `L8`, `L12`, `L16`, or `L24` for every camera of a temporal sample.
2. A token-level Mixture-of-Depths (MoD) router that routes a fixed 50% capacity of tokens through selected later EVA blocks and bypasses the remaining tokens through residual paths.

Both routers are trained to minimize detection loss subject to a compute budget. They do not use current-frame ground truth, a drift/similarity prediction target, or offline metadata as primary inputs.

## Scope and Compatibility

- Preserve exact baseline behavior when `temporal_memory.enabled=False`.
- Preserve the full image-encoder behavior when `keyframe_interval=1` or a router selects `L24`.
- Keep the controller, memory updater, and adapter interfaces detector-agnostic. RepDETR3D EVA-02 is the first integration; StreamPETR remains supported at the interface level and is not hard-coded out.
- Store/update pre-PE appearance memory and let the existing detector path recompute current-frame PE.
- Every router decision is unified across all cameras of one temporal sample. Router predictions remain per sample: a training batch with four samples produces four independent action distributions and may execute grouped depth branches. Router evaluation/test configs use `samples_per_gpu=1`, matching the intended latency-critical single-sample inference path.
- Reset conditions already handled by `TemporalMemoryController` remain authoritative: scene boundary, discontinuity, missing previous frame, and configured maximum frames since a full path.
- Both primary router variants set `max_frames_between_full=5`, forcing a full `L24` encoding at least every fifth frame.

## Shared Router Inputs

The deployable input vector at frame `t` is composed only of online quantities:

- A pooled representation of the current EVA block-8 feature, computed for every candidate-routed sample.
- A pooled representation of the cached pre-PE appearance memory from `t-1`; use zeros plus a validity bit when memory is unavailable.
- Online temporal/ego metadata already available to the controller: frame index/age since full path, timestamp delta, ego translation magnitude, rotation angle, speed, and yaw-rate when present.

The router does not consume current-frame annotations, object counts, ground-truth boxes, oracle difficulty labels, or a learned target that predicts representation drift. A current-L8/previous-memory interaction is allowed only as a direct input to the decision network; it is not supervised separately.

Normalize scalar metadata through registered buffers from training-set statistics. Pool visual tensors by camera-aware spatial mean followed by mean across cameras, then use small learned projections before concatenation with metadata.

## A. Four-Exit Depth Router

### Behavior

All cameras and all tokens of a sample follow one selected depth:

| Action | Encoder work | Detector feature path |
|---|---|---|
| `L8` | blocks 1--8 | existing layer-specific partial projector -> temporal updater |
| `L12` | blocks 1--12 | existing layer-specific partial projector -> temporal updater |
| `L16` | blocks 1--16 | existing layer-specific partial projector -> temporal updater |
| `L24` | full blocks 1--24 | existing full detector feature path and memory refresh |

The first 8 blocks are always evaluated for non-keyframe candidates because `L8` is the shallowest exit and supplies the current online visual signal. `L24` remains mandatory for first frames, controller resets, and the reset-5 condition. Training preserves independent per-sample decisions even when several samples happen to select different exits; it groups equal-depth samples only as an execution optimization. Evaluation/test runs one sample per GPU, so a selected partial exit avoids deeper encoder work without cross-sample batching constraints.

`DepthExitRouter` is a LayerNorm -> bottleneck MLP -> four-logit projection. It receives the shared inputs above and produces a categorical distribution over the four actions. During training, a straight-through Gumbel-Softmax one-hot action carries gradients through the logits; the selected branch is still executed as a hard branch. During evaluation, `argmax` chooses one action deterministically. Distributed execution gathers samples by selected action, runs only the needed additional blocks for each group, restores original sample ordering, and then passes the resulting adapted partial features to the existing temporal controller/updater.

### Adapters

Keep the established layer-specific `PartialFeatureProjector` paths unchanged for `L8`, `L12`, and `L16`; the existing `LayeredPartialFeatureProjector` selects the matching learned projector by action. `L24` uses the existing full neck/feature path. The depth router adds no new exit adapter and has no adapter pretraining stage.

### Optimization and Budget

For a batch of routing decisions with categorical probabilities `p`, expected normalized encoder cost is:

`E[c] = mean(sum_a p[a] * c[a])`, where `c = [8, 12, 16, 24] / 24`.

The training objective is:

`L_total = L_detection + lambda_budget * (E[c] - target_cost)^2 + lambda_balance * L_balance`.

The primary target is `target_cost=0.50`; the forced `L24` reset frames are included in actual latency accounting and can either be included in the moving expected-cost statistic or exempted by a documented `budget_on_routable_frames` flag. The default is routable-frame budgeting so the controllable router target remains meaningful.

`L_balance` is a short warm-up regularizer that prevents immediate all-`L24` or all-`L8` collapse. It matches the mean action probability to a configurable prior whose expected cost is 0.50. It decays to zero after a configurable warm-up fraction; thereafter only detection loss and the expected-cost penalty determine routing. Log action fractions, expected cost, realized cost, forced-L24 count, and budget loss.

### Initial Experiments

1. Frozen EVA-02; train router, existing layer-specific partial projectors, gated updater, and detector head using detection plus budget losses.
2. Same as (1), with a different balance-warm-up length.
3. Same as (1), target normalized compute 0.50 and reset-5, with `lambda_balance=0` after warm-up.

The primary comparison is against the existing four-level MLP policy with reset-5, but the new rows must be named as detection-loss/budget routers rather than difficulty or drift policies.

## B. Token-Level Mixture-of-Depths Router

### Backbone Placement and Capacity

Keep blocks 1--8 dense. EVA-02 alternates window-attention and global-attention blocks; arbitrary packed-token routing is only well-defined for the existing global-attention implementation. Apply MoD routing at the later global blocks `9, 12, 15, 18, 21` (one-indexed). The intervening window blocks and the final block 24 remain dense. This gives every selected routing block a later full-compute recovery block while retaining the pretrained positional/window behavior.

At every routed block and for every camera independently, score all spatial tokens and select exactly the top 50% for the expensive block path. The same capacity ratio is enforced for every camera and temporal sample, yielding a unified predictable frame budget without camera-specific depth choices. Non-selected tokens bypass the heavy block through its residual connection. Top-k selection therefore fixes the routed capacity exactly; no learned drift or budget-prediction target is used.

The selected tokens are packed to a static `ceil(0.5 * token_count)` capacity, processed through the block attention and MLP, then scattered back. The skipped tokens preserve their incoming residual-state values. The router score multiplies the selected block update before scatter so detection loss has a gradient path to score magnitude, as in MoD. The hard top-k identity is used for forward execution. A straight-through soft surrogate may be used only to stabilize score gradients, never to change the fixed forward capacity.

### Per-Junction Components

Each routed junction owns independent modules:

- `LayerNorm` over the current token state before routing.
- `TokenDepthRouter`: a bottleneck MLP ending in one scalar score per token.
- `TokenDepthAdapter`: a small residual adapter applied after the routed block update before scatter.

This follows the N-vium principle that a pretrained intermediate representation needs a dedicated, depth-specific alignment module. It also limits the adaptation surface when the EVA backbone is frozen. Dense blocks do not get new routers or adapters.

### Training Regimes

Run two matched regimes with the same routed layers, 50% capacity, reset-5 controller policy, updater, optimizer schedule, and detection evaluation protocol:

1. **Frozen-backbone MoD:** Freeze EVA-02 blocks and patch embedding. Train the eight token routers, eight token adapters, partial projector, temporal updater, and detector head. Optionally pretrain adapters against the full path before joint detector training.
2. **N-vium-style joint MoD:** Initialize from the same RepDETR3D checkpoint, then train the EVA backbone together with routers, adapters, projector, updater, and detector head. Use a router warm-up phase with a uniform routing-score/use regularizer, then switch to detection loss plus the fixed-capacity MoD forward path. The warm-up is a collapse mitigation rather than a drift objective.

Because capacity is exactly 50% at every routed block, the principal optimization objective is detection loss. Report the constant theoretical routed capacity and actual wall-clock latency rather than claiming a learned variable compute budget.

### Initial Experiments

1. Frozen-backbone MoD, 50% capacity, no adapter pretraining.
2. Frozen-backbone MoD, 50% capacity, adapter warm-up.
3. Joint N-vium-style MoD, 50% capacity, router warm-up.

Do not add lower-capacity experiments until the 50% versions are stable. Later ablations may compare 25% and 12.5% capacity, but must retain the same routed-layer placement and report measured latency, GPU memory, mAP, and NDS.

## Training, Measurement, and Tests

### Training

- Keep full encoder teacher distillation optional for updater/adapter initialization, but do not use it as router supervision.
- New configs load `ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth`; baseline configs remain unmodified.
- Router training configs keep the configured per-GPU training batch size and emit one route per sample. Router evaluation/test configs set `data.samples_per_gpu=1` so reported latency reflects unbatched per-sample routing.
- Record whether EVA is frozen, whether adapters were warmed up, budget target, balance warm-up length, routed blocks, capacity, reset interval, and all trainable parameter groups.

### Latency

Profile CUDA-synchronized per-frame time separately for L8 probe, routed backbone blocks, exit adapters/projector, memory updater, detector head, and total. For the four-exit router, include L8 in every non-keyframe decision and include forced L24 reset frames in aggregate sequence latency. For MoD, report fixed 50% capacity plus measured wall-clock latency; top-k, packing, scattering, adapter, and router overhead must be included.

### Required Tests

- Disabled temporal memory matches baseline numerically.
- Router plus `keyframe_interval=1` follows the full path.
- Reset-5 forces L24 even if router logits prefer a partial action.
- Depth router selects exactly one action per sample, shares it across cameras, restores batched order after grouped execution, and calculates expected cost from soft probabilities.
- No current GT/offline annotation key is accepted by deployable router feature construction.
- Budget and warm-up losses yield finite gradients into depth-router parameters.
- MoD top-k selects exactly 50% capacity per camera and preserves original dense token geometry after scatter.
- MoD skipped tokens take residual bypass; selected tokens receive routed block plus adapter update.
- MoD router and adapters receive finite detection gradients in frozen and joint modes.
- Existing temporal cache, scene-boundary, discontinuity, PE identity, and shape tests continue to pass.

## Results Table Entries

Append rows to `RESULTS.md`; leave metric columns blank until each run completes. Add at least these rows:

1. `RepDETR3D EVA02 depth router L8/L12/L16/L24 detection+budget target50 reset5 frozen EVA`.
2. `RepDETR3D EVA02 depth router L8/L12/L16/L24 detection+budget target50 reset5 frozen EVA long balance warm-up`.
3. `RepDETR3D EVA02 MoD global blocks 9-12-15-18-21 capacity50 frozen EVA`.
4. `RepDETR3D EVA02 MoD global blocks 9-12-15-18-21 capacity50 frozen EVA adapter warm-up`.
5. `RepDETR3D EVA02 MoD global blocks 9-12-15-18-21 capacity50 N-vium-style joint training`.

Each row must state that inputs are online-only, no GT and no drift target are used, and list the exact train/test command. The MoD rows must identify frozen versus joint backbone as a controlled experimental difference.

## Out of Scope

- Per-camera depth decisions.
- Offline/current GT metadata in deployable routes.
- A router supervised to predict feature drift, similarity, or an oracle depth label.
- Lower MoD capacities before the 50% implementation is verified.
- Altering the existing baseline configs or replacing the detector adapter abstraction.
