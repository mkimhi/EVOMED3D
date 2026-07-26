# Dynamic Policy Plan

## Goal

Build a dynamic frame/layer policy for RepDETR3D EVA-02 temporal memory that
maximizes mAP/NDS under a target latency budget. The policy chooses one action
per frame:

```text
L8   = run EVA block 8 partial path and update memory
L12  = run EVA block 12 partial path and update memory
L16  = run EVA block 16 partial path and update memory
L24  = run the full encoder path
```

`L24` is the full-feature teacher/reference path and is always required on the
first frame of a scene, scene discontinuities, and any forced reset.

The first target should be a K=3-equivalent average budget, because current
fixed K=3 L8 gated residual results are the strongest learned-update reference.
The policy should be evaluated by actual wall-clock latency, not FLOPs alone.

## Rules For The Research Path

Test each cue by itself before combining cues. Even when two cues logically
belong together, isolate the effect first so the result table can explain which
signal actually improves the accuracy/latency tradeoff.

Do not use the summed training loss as the oracle target. The total loss mixes
classification, box regression, denoising, encoder/2D auxiliary losses, and
hand-chosen weights. It is not a reliable per-frame proxy for mAP or NDS.

For oracle-style labels, use the cheapest action that preserves the full L24
detector output. Feature agreement is useful as an auxiliary signal, but the
main oracle should be based on detection agreement with L24.

## Baseline References

Before dynamic policies, keep these fixed policies as anchors:

| policy | purpose |
|---|---|
| Full L24 every frame | Accuracy and latency upper reference |
| Naive reuse K=2/K=3 | Simple skip baseline |
| Partial-only L8/L12/L16 | Tests current partial features without memory |
| Gated residual K=3 L8 | Current best learned memory-update reference |
| Gated residual K=3 L8 + distill | Tests whether teacher feature loss helps |

All dynamic policies should report against the same columns:

```text
method | actions | mAP | NDS | fps | inference time | full-frame rate | avg action cost | action histogram
```

Minimum logging for every policy:

- number and fraction of frames routed to each of `L8/L12/L16/L24`
- average frames between `L24` calls
- max frames between `L24` calls
- per-scene action histogram
- actual latency with `model.tm_debug_latency=True`
- GPU memory

## Stage 1: Ego Speed Only

This is the first dynamic baseline because it is cheap, deterministic, and uses
metadata already available in the nuScenes samples.

Policy:

```text
if scene boundary or first frame:
    L24
elif ego_speed < 2 m/s:
    L8
elif ego_speed < 8 m/s:
    L12
elif ego_speed < 12 m/s:
    L16
else:
    L24
```

Expected behavior:

- slow or stopped scenes should spend mostly `L8`
- high-speed ego motion should escalate to deeper features or full encoding
- the policy should beat fixed K=3 reuse and approach fixed K=3 gated residual
  under similar or better latency

Run ablations:

| name | thresholds |
|---|---|
| conservative speed | `1, 5, 10 m/s` |
| default speed | `2, 8, 12 m/s` |
| aggressive speed | `3, 10, 15 m/s` |

## Stage 2: Ego Yaw Only

Test yaw rate separately from speed. Turning can stale camera-view memory even
when translation speed is low.

Policy:

```text
if scene boundary or first frame:
    L24
elif abs(yaw_rate) < 3 deg/s:
    L8
elif abs(yaw_rate) < 7 deg/s:
    L12
elif abs(yaw_rate) < 12 deg/s:
    L16
else:
    L24
```

Run ablations:

| name | thresholds |
|---|---|
| conservative yaw | `2, 5, 10 deg/s` |
| default yaw | `3, 7, 12 deg/s` |
| aggressive yaw | `5, 10, 15 deg/s` |

Evaluate whether yaw-only improves classes sensitive to view change, especially
near-camera objects and objects entering/exiting camera overlap regions.

## Stage 3: Ego Speed + Yaw

Combine ego speed and yaw only after the separate speed-only and yaw-only
experiments are logged.

Policy:

```text
speed_action = speed_policy(ego_speed)
yaw_action = yaw_policy(abs(yaw_rate))
action = deeper(speed_action, yaw_action)
```

Depth order:

```text
L8 < L12 < L16 < L24
```

This policy should answer whether simple ego motion can allocate full encoding
to harder frames without needing learned confidence or feature drift.

## Stage 4: Time-Since-Full Safety Rule

Test max-gap safety separately. Do not silently add it to ego policies before
the no-safety variants are evaluated.

Policy:

```text
action = base_policy(...)
if frames_since_L24 >= max_full_gap:
    action = L24
```

Run:

```text
max_full_gap in [3, 5, 8]
```

This isolates whether forced full refresh is necessary to prevent memory drift.

## Stage 5: Detection Confidence Signals

Confidence should not be a single raw score. Test signals separately, then
combine only if they individually correlate with failure.

Candidate signals:

| signal | definition | failure indication |
|---|---|---|
| top-k score | mean score of top-k detections | low score |
| class margin | top class score minus second class score | low margin |
| class entropy | entropy over class probabilities | high entropy |
| unmatched boxes | high-score boxes not matched to previous frame | high count |
| temporal box jump | matched boxes with large center/yaw/velocity jump | high jump |

Recommended first confidence baseline:

```text
if mean_top20_margin < threshold:
    escalate one level
elif temporal_instability > threshold:
    escalate one level
else:
    keep ego-policy action
```

Test confidence on top of the best ego-only policy only after standalone
confidence diagnostics are logged.

## Stage 6: Feature Drift Signals

Feature drift should come after ego policies and confidence diagnostics. It is
closer to the temporal-memory mechanism, but it depends on projected features
being calibrated.

Candidate drift metrics:

```text
drift_l8 = 1 - cosine(Project(L8_t), previous_memory)
delta_norm = ||Project(L8_t) - previous_memory||
updater_gate_mean = mean(gate)
updater_residual_norm = ||updated_memory - previous_memory||
```

Initial drift policy:

```text
if drift_l8 < low_threshold:
    L8
elif drift_l8 < medium_threshold:
    L12
elif drift_l8 < high_threshold:
    L16
else:
    L24
```

Feature drift must be evaluated both globally and per camera. Per-camera drift
may reveal view-specific staleness that a single global score hides.

## Stage 7: Oracle Labels

Oracle labels should answer:

```text
What is the cheapest action that preserves the full L24 detector output?
```

For each sampled frame, run:

```text
L8, L12, L16, L24
```

Then compare each candidate to L24 detections.

Detection agreement metrics:

- match candidate boxes to L24 boxes by class and center distance
- missed high-confidence L24 boxes
- extra high-confidence candidate boxes
- center/size/yaw/velocity error for matched boxes
- class score difference for matched boxes

Oracle label:

```text
if L8 agrees with L24: L8
elif L12 agrees with L24: L12
elif L16 agrees with L24: L16
else: L24
```

Store both the chosen label and the agreement gap for every action. This allows
future training as either action classification or expected-regret regression.

Do not use total detection training loss as the oracle label.

## Stage 8: Learned Policy

Only train a learned policy after the simple cue ablations identify useful
signals.

Initial policy inputs:

```text
ego_speed
abs_yaw_rate
time_since_L24
optional confidence signal
optional feature drift signal
```

Start with an imitation policy trained on oracle labels. Then add budget-aware
fine-tuning only if imitation is stable.

Budget control:

```text
target_average_cost = K3_equivalent
budget_balance += target_average_cost - chosen_action_cost
discourage L24 when budget_balance is low
allow L24 when budget_balance is high
```

End-to-end Gumbel over `L8/L12/L16/L24` should be treated as a later ablation,
not the first dynamic-policy method.

## Acceptance Criteria

A dynamic policy is useful only if it satisfies all of:

- improves mAP/NDS over naive reuse at comparable latency
- matches or improves fixed K=3 L8 gated residual accuracy/latency tradeoff
- reports actual wall-clock latency and action histogram
- does not use `L24` so often that the speedup disappears
- preserves baseline behavior when temporal memory is disabled

Stop or redesign if:

- dynamic routing mostly collapses to `L24`
- dynamic routing mostly collapses to `L8` and loses accuracy
- confidence or drift signals do not correlate with failure cases
- policy overhead erases the encoder savings
