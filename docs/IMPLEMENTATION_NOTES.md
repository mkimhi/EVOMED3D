# Implementation notes

## Coding priorities

1. Do not break the baseline.
2. Add config flags for every new behavior.
3. Keep temporal memory modules isolated from detector-specific code.
4. Use small commits/patches.
5. Add shape assertions early.

## Suggested file organization

Use the repository's existing module conventions. If no better location exists, create:

```text
projects/mmdet3d_plugin/models/temporal_memory/
    __init__.py
    controller.py
    detector_adapters.py
    memory_packet.py
    updaters.py
    pe_handler.py
    distillation.py
    schedulers.py
```

Detector-specific adapters can live in:

```text
projects/mmdet3d_plugin/models/temporal_memory/adapters/
    repdetr3d_adapter.py
    streampetr_adapter.py
```

## Config flags

Add flags like:

```python
temporal_memory = dict(
    enabled=False,
    detector_adapter="repdetr3d",
    keyframe_interval=3,
    updater="gated_residual",
    partial_layer="middle",
    memory_is_pre_pe=True,
    pe_mode="separate_recompute",
    use_ego_conditioning=True,
    use_distillation=True,
    distill_weight=1.0,
    train_with_teacher=True,
    inference_teacher_forbidden=True,
    policy=dict(
        type="fixed_interval",  # fixed_interval, cycle, or mlp
        partial_action="L8",
    ),
)
```

Learned MLP policy configs may either load exported weights:

```python
policy=dict(
    type="mlp",
    checkpoint="outputs/policy_signal_stats/val/mlp_policy_online.pt",
    fallback_action="L24",
)
```

or intentionally skip loading weights for a train-from-scratch/random-policy ablation:

```python
policy=dict(
    type="mlp",
    checkpoint=None,
    load_weights=False,
    feature_names=["frame_number", "dt_s", "ego_speed_mps"],
    hidden_dim=32,
    labels=["easy", "medium", "hard"],
    actions=dict(easy="L8", medium="L12", hard="L24"),
)
```

## Debug flags

```python
tm_debug_shapes=True
tm_debug_latency=True
tm_debug_compare_full_path=False
tm_assert_no_teacher_in_inference=True
```

## Feature extraction

For EVA-style ViTs, implement intermediate feature collection carefully:

- Avoid storing every block output by default.
- Collect only requested layers.
- Use hooks only if they are robust under distributed training and checkpointing.
- Prefer explicit forward options if the backbone supports them.

## Memory cache

The memory cache should reset on:

- new scene
- sequence discontinuity
- missing previous frame
- large timestamp gap
- explicit batch boundary

The cache key should include scene/sample identity when batch size > 1.

## Batch handling

Do not assume batch size 1 unless the config requires it. If batch sequences are interleaved, memory must be keyed per sequence.

Simpler first implementation may assert:

```python
assert batch_size == 1
```

but this must be clearly documented.

## Latency measurement

Use CUDA synchronization around measured regions:

```python
torch.cuda.synchronize()
start = time.perf_counter()
...
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Report:

- full backbone time
- partial backbone time
- updater time
- detector head time
- total per-frame time

## Reproducibility

Log:

- config path
- checkpoint path
- git commit
- keyframe interval
- updater type
- partial layer
- PE mode
- distillation weight
- whether backbone/head were frozen

## Common failure modes

### Shape mismatch

Fix using adapter conversions, not ad-hoc reshapes in the updater.

### Stale PE

If outputs degrade badly on non-keyframes, verify whether memory includes PE. Try pre-PE memory and recomputed PE.

### No speedup

Profile updater and partial feature extraction. Cross-attention may be too expensive.

### Distillation improves but detection does not

The distilled feature may match globally but lose object-critical detail. Add foreground-weighted feature distillation only after the simple loss is tested.

### Training collapse

Reduce update magnitude:

```text
M_t = M_{t-1} + alpha * sigmoid(gate) * residual
```

Initialize `alpha` small, e.g. 0.1.
