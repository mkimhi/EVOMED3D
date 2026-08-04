# Drift Quantile versus Depth Degradation Notebook

## Objective

Create a reproducible analysis notebook that tests whether adjacent-frame drift in the full L24 representation identifies frames where shallow encoder depth causes larger detection degradation. This figure should support the paper's compute-allocation premise with task-level evidence rather than representation predictability alone.

## Scope

Add `notebooks/drift_quantile_vs_depth_degradation.ipynb`. The notebook analyzes aligned, precomputed counterfactual predictions for L8, L12, L16, and L24. It does not run detector inference. Each depth must have been evaluated from the same pre-frame temporal-memory state; the notebook validates that requirement through an input manifest.

The notebook uses:

- `outputs/policy_signal_stats/val/policy_feature_similarity.csv` for adjacent-frame L24 cosine drift;
- one official nuScenes-format `results_nusc.json` artifact for each of L8, L12, L16, and L24;
- nuScenes validation ground truth;
- an optional aligned per-frame detector-loss table.

The manifest is JSON with `shared_pre_frame_state: true`, a common run identifier, the canonical state-advance action, and the four prediction paths. Each prediction artifact must contain exactly the same sample-token set. The loss table, when present, has one row per token and columns `loss_L8`, `loss_L12`, `loss_L16`, and `loss_L24`.

## Scientific Endpoint

For frame `t` and candidate depth `l`, define

```text
drift_t = 1 - similarity_L24_mean_t
degradation_t_l = error_t_l - error_t_L24
```

The primary per-frame error is computed by class-aware Hungarian matching between predictions and ground truth. Predictions and ground-truth boxes from different classes cannot match. Same-class pairs beyond a configurable center-distance gate cannot match. The default endpoint averages the following normalized costs over matched and unmatched instances:

- confidence error (`1 - prediction score`);
- 3D center displacement divided by the matching distance gate;
- mean absolute log-size ratio;
- wrapped absolute yaw error divided by pi;
- velocity-vector error divided by a configurable velocity scale;
- unmatched-prediction false-positive penalty;
- unmatched-ground-truth false-negative penalty.

All component weights, the prediction-score floor, and matching thresholds live in one configuration cell and are recorded with the outputs. The default primary endpoint gives equal aggregate weight to confidence and geometry, with false-positive and false-negative penalties of one; the exact component values remain editable rather than hidden in helper code. A box-only version excludes confidence error but retains geometry and unmatched-box penalties. When supplied, detector-loss degradation is analyzed as a second sensitivity endpoint.

The combined matched-prediction endpoint is a per-frame surrogate, not nuScenes mAP or NDS. The notebook and figure captions must state this explicitly.

## Analysis Flow

1. Configure input and output paths, matching thresholds, component weights, drift quantiles, bootstrap seed, and bootstrap repetitions.
2. Load the L24 similarity table, prediction artifacts, ground truth, optional detector losses, and counterfactual-run manifest.
3. Validate unique tokens, identical depth coverage, scene identity, finite drift values, and shared-pre-state provenance.
4. Remove scene starts because adjacent-frame drift is undefined.
5. Match each depth's predictions to ground truth and calculate combined and box-only per-frame error.
6. Subtract the paired L24 error from each shallow-depth error.
7. Assign frames to five equal-frequency quantiles using the empirical L24 drift distribution.
8. Calculate per-quantile means and scene-block-bootstrap 95% confidence intervals using 2,000 seeded resamples by default.
9. Calculate frame-level Spearman correlations for L8, L12, and L16, with the same scene-block-bootstrap confidence intervals.
10. Export plots, statistics, configuration, and sample-count summaries.

## Figures

### Primary Quantile Figure

Plot L24 drift quantiles on the x-axis and mean matched-prediction degradation on the y-axis. Draw separate L8, L12, and L16 curves with scene-bootstrap 95% confidence intervals. Include a zero-degradation reference line and report the three Spearman correlations in the legend or adjacent statistics table.

### Alternative Scatter Figure

Create one subplot for each of L8, L12, and L16. Plot transparent, rasterized per-frame points of continuous L24 drift against paired degradation. Overlay the quantile means and a seeded isotonic-regression trend. This view exposes dispersion that the quantile summary hides.

### Sensitivity Views

Repeat the quantile analysis for box-only degradation. If the optional detector-loss table exists, repeat it for detector-loss degradation; otherwise mark that section as skipped without failing the notebook.

## Outputs

Write publication PDFs to:

- `paper/figures/fig_drift_vs_depth_degradation.pdf`
- `paper/figures/fig_drift_vs_depth_degradation_scatter.pdf`

Write PNG previews and machine-readable artifacts under `outputs/drift_depth_analysis/`, including:

- per-frame degradation CSV;
- per-quantile summary CSV;
- correlation summary CSV;
- resolved analysis configuration JSON;
- validation/sample-count summary JSON.

## Validation and Failure Behavior

The notebook stops with a clear error when:

- a required input is missing;
- tokens are duplicated or depth artifacts cover different frames;
- prediction tokens cannot be joined to drift or ground truth;
- the manifest does not confirm the same pre-frame memory state across depth branches;
- fewer than two scenes or too few distinct drift values remain for uncertainty estimates.

Optional detector-loss analysis is the only section allowed to skip cleanly.

The final notebook cell checks that expected figures and statistics files exist and are non-empty. A lightweight synthetic-data path exercises joining, matching, degradation, quantile aggregation, and plotting without requiring nuScenes or a GPU.

## Non-goals

- Running four detector branches inside the notebook.
- Claiming that the surrogate is a decomposition of mAP or NDS.
- Comparing independently accumulated temporal rollouts as if they were paired counterfactuals.
- Adding a reusable analysis framework or new dependency solely for this figure.
