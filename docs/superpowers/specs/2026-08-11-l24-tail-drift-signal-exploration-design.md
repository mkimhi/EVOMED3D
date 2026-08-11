# L24 tail-drift signal exploration design

## Goal

Build an exploratory analysis that tests whether sparse or camera-localized L24 representation changes predict naive-reuse risk better than the current global mean L24 drift. The immediate purpose is signal selection: decide which drift statistic should be used in the next paper-style degradation figure.

## Motivation

The current global mean L24 cosine drift has only weak association with absolute GT loss. This is plausible because most image tokens across six cameras are static or background-like, so averaging over all tokens can dilute a small but important representation change. A policy signal should answer a risk-oriented question: did any meaningful tail of detector-consumed L24 tokens change enough that naive reuse becomes unsafe?

## Scope

Add a new exploratory notebook:

```text
notebooks/policy_l24_drift_signal_explore.ipynb
```

The notebook compares compact L24 drift signals against the already collected naive-reuse absolute GT losses. It is exploratory and should not be the final paper figure. After signal selection, a later production script/notebook will reuse the selected signal for the depth-degradation or reuse-risk figure.

## Required collector extension

The existing collection path saves `full_l24_drift_statistics.json`, but that file only contains per-level cosine sums and vector counts. Tail metrics require token-wise cosine distributions. The collector will therefore be extended to write compact per-target drift summaries, not full tensors.

For each selected target and each available offset (`dt1`, `dt2`), compute token-wise cosine drift:

```text
token_drift = 1 - cosine(feature_t, feature_t-k)
```

The implementation will flatten all non-channel dimensions into tokens while using the channel dimension for cosine similarity, matching the existing L24 cosine definition. It will save only scalar summaries.

## Candidate signals

Global token-tail signals:

```text
mean_drift
p90_token_drift
p95_token_drift
p99_token_drift
top1_mean_token_drift
top5_mean_token_drift
top10_mean_token_drift
max_token_drift
```

Camera-aware signals, when the feature tensor keeps a camera/view dimension:

```text
max_camera_mean_drift
max_camera_p95_drift
max_camera_top5_mean_drift
camera_drift_imbalance
```

`camera_drift_imbalance` is defined as:

```text
max_camera_top5_mean_drift / mean_camera_top5_mean_drift
```

with finite handling for zero denominators.

If the feature layout does not expose a reliable camera dimension, camera-aware metrics will be omitted with explicit provenance in the summary rather than guessed.

## Exploratory analysis outputs

The notebook will write outputs under:

```text
outputs/l24_drift_signal_explore/
```

Expected files:

```text
signal_frame.csv
signal_correlation_summary.csv
signal_quintile_summary.csv
signal_rank_prediction_loss.png
signal_rank_box_loss.png
top_signals_quintile_grid.png
explore_config.json
validation_summary.json
```

The ranked summary will include:

```text
offset
signal
outcome
spearman
ci_low
ci_high
top_vs_bottom_quintile_gap
complete_frames
```

## Decision rule

Prefer the signal with the strongest and most stable association with `prediction_loss`, provided it remains interpretable. Do not choose raw `max_token_drift` as the headline signal unless it substantially outperforms tail-mean and percentile metrics, because it is easier to dismiss as noise-sensitive.

Expected strongest candidates:

```text
top5_mean_token_drift
max_camera_top5_mean_drift
p95_token_drift
```

## Interpretation boundary

This analysis remains an association study. It may show that tail L24 drift is a better compute-allocation signal than mean drift, but it does not by itself prove causal degradation from reuse. The later production degradation figure should phrase the result as risk prediction unless it includes an explicit full-vs-reuse counterfactual.

## Validation

Unit tests will cover:

- token-tail drift summaries on synthetic tensors;
- missing `dt2` behavior for the first K=2 reuse frame;
- ranking/correlation summaries with `NaN` handling;
- camera-aware metrics only when the feature layout supports them.

Notebook smoke execution will use deterministic synthetic data and verify that all expected output files are created.
