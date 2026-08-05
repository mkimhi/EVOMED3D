# L24 drift versus naive-reuse risk analysis

## Objective

Test the policy-motivation hypothesis for RepDETR3D EVA-02: when the current
full L24 image representation differs substantially from recent full L24
representations, the deployed naive-reuse K=2 detector has greater detection
error against nuScenes ground truth.

This is a base-model analysis. It must not use L8, L12, L16, or any
partial-depth representation as a signal or prediction source.

## Experimental unit and sampling

The analysis set contains exactly 2,000 target frames: 20 deterministic,
uniformly spaced eligible reuse frames from each of 100 deterministic
validation scenes. An eligible target has two preceding frames in the same
scene and is a non-keyframe under the K=2 schedule. Keyframes are excluded:
they do not exercise naive reuse and are not policy-intervention opportunities.

The collector will record the selected scene and target sample tokens in a
manifest. Re-running with the same seed, dataset version, configuration, and
scene selection must select the same 2,000 targets.

## Collection

Use the checkpoint and the temporal reuse configuration from the `RepDETR3D
EVA02 naive reuse K=2` row in `RESULTS.md`:

```text
projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_reuse_k3.py
ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth
model.temporal_memory.keyframe_interval=2
```

For every target and its two preceding same-scene samples, the collector runs
the full base L24 encoder and stores a compact, token-count-aware full-L24
representation summary sufficient to recompute the defined drift metric. It
runs the K=2 reuse path sequentially and writes only the K=2 target-frame
predictions in official nuScenes `results_nusc.json` format. The collection
manifest records config/checkpoint hashes, selected tokens, output paths,
dataset version, seed, and that the target predictions came from the K=2 reuse
path.

## Metrics

For target frame `t`, representation drift is the cosine distance between
full-L24 representations:

```text
drift_dt1(t) = 1 - cosine(L24(t), L24(t-1))
drift_dt2(t) = 1 - cosine(L24(t), L24(t-2))
```

Per-frame K=2 error is measured against nuScenes annotations using the
existing class-aware, center-gated Hungarian matching endpoint. It has two
pre-specified outcomes:

1. **Prediction loss:** confidence, center, size, yaw, velocity, and explicit
   false-positive/false-negative penalties.
2. **Box loss:** center, size, yaw, velocity, and explicit unmatched-box
   penalties, with confidence removed.

These are absolute K=2 errors against ground truth, not K=2-minus-full
degradation. They characterize deployment risk and may include frame-intrinsic
detection difficulty; they do not by themselves estimate the causal penalty of
reuse.

## Statistics and figures

For each offset/outcome pair, report Spearman correlation and a 95% confidence
interval from bootstrap resampling complete scenes. Do not resample individual
frames. Also divide drift into five equal-frequency bins and report mean loss
and scene-bootstrap 95% confidence intervals per bin.

Create one publication-ready 2-by-2 figure:

| Rows | Columns |
| --- | --- |
| prediction loss, box loss | delta t=1, delta t=2 |

Each panel contains a density-aware scatter plot, an isotonic trend, quantile
means with uncertainty bars, a zero-free informative axis range, and an in-panel
Spearman estimate with its scene-bootstrap interval. Save PDF and 300-dpi PNG
versions in `outputs/l24_drift_reuse_risk/` and `paper/figures/`.

The notebook writes the per-frame table, summary tables, resolved analysis
configuration, validation manifest, and figures. It fails instead of silently
substituting synthetic data when required artifacts are absent or token coverage
is inconsistent.

## Validation

Unit tests will cover deterministic balanced scene/frame selection, exclusion
of keyframes and cross-scene offsets, drift computation (including identity),
metric joins with complete token coverage, and scene-bootstrap output shape.
The collector must reject a selection that cannot provide 100 scenes with 20
eligible targets each, and the notebook must reject malformed manifests or
incomplete prediction coverage.
