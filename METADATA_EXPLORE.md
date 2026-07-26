# Metadata and Feature-Similarity Exploration Plan

This note summarizes the current policy-feature tables and the first diagnostic
results for learning dynamic L8/L12/L16/L24 policies.

## Available Tables

### Metadata features

File:

```text
outputs/policy_signal_stats/val/policy_signal_stats.csv
```

Rows: `6019` validation frames.

This table contains frame-level metadata and GT-derived analysis features:

- frame/time: `frame_number`, `info_frame_idx`, `is_scene_start`, `dt_s`
- ego motion: `ego_x_m`, `ego_y_m`, `ego_z_m`, `ego_yaw_deg`,
  `delta_ego_translation_m`, `ego_speed_mps`, `ego_accel_mps2`,
  `yaw_rate_deg_s`, `abs_yaw_rate_deg_s`, `yaw_accel_deg_s2`,
  `abs_yaw_accel_deg_s2`
- object density/range: `num_annotations`, average/median/min/p10/p90 object
  distance, near-object counts at 10/20/30/50 meters
- box geometry: average length/width/height/area/volume
- object motion: average and max GT object speed from `gt_velocity`
- image-set density: total/max/min/average camera annotation count,
  average/max 2D box area
- new-object proxy: `new_object_proxy_count`, `matched_object_proxy_count`,
  `disappeared_object_proxy_count`, `new_object_proxy_fraction`

Caveat: the nuScenes temporal pickle does not include stable instance tokens.
The new-object fields are proxies produced by matching same-class boxes to the
previous frame in global XY within a distance threshold. They are useful for
analysis, but they are not true instance IDs.

### Representation consistency features

File:

```text
outputs/policy_signal_stats/val/policy_feature_similarity.csv
```

Rows: `6019` validation frames. `5869` rows have previous-frame similarity;
scene starts have `has_previous_frame=0` and similarity values are missing.

This table contains adjacent-frame raw EVA feature cosine similarities:

- `similarity_L8_<CAM>` for 6 cameras
- `similarity_L12_<CAM>` for 6 cameras
- `similarity_L16_<CAM>` for 6 cameras
- `similarity_L24_<CAM>` for 6 cameras
- `similarity_L8_mean`, `similarity_L12_mean`, `similarity_L16_mean`,
  `similarity_L24_mean`

`L24` here means the final EVA block before SimpleFPN. These are raw backbone
feature similarities, not detector/FPN-interface similarities.

The two tables join cleanly by `token`.

## Working Objective

The final policy objective should not merely be predicting similarity. The
policy objective should be:

```text
Choose the cheapest action L8/L12/L16/L24 that keeps representation/detection
change below an acceptable threshold while preserving mAP/NDS.
```

Similarity is a practical first proxy because it is cheap to label offline and
can be measured for all frames. Later, detection agreement with full L24 should
replace or complement similarity as the oracle target.

## Similarity Target Distribution

Initial diagnostics used `similarity_L24_mean` as the target.

For rows with a previous frame (`n = 5869`):

```text
min    ~= 0.49
p25    ~= 0.67
median ~= 0.72
p75    ~= 0.77
p95    ~= 0.93
max    ~= 0.96
```

The target is broad enough to be useful. It is not a nearly constant signal.

## Quick Correlation Diagnostics

The strongest metadata correlations with lower `similarity_L24_mean` were:

```text
delta_ego_translation_m        Spearman ~= -0.553
ego_speed_mps                  Spearman ~= -0.550
abs_yaw_rate_deg_s             Spearman ~= -0.447
abs_yaw_accel_deg_s2           Spearman ~= -0.435
disappeared_object_proxy_count Spearman ~= -0.404
new_object_proxy_count         Spearman ~= -0.394
near_object_count_50m          Spearman ~= -0.292
near_object_count_30m          Spearman ~= -0.285
num_annotations                Spearman ~= -0.279
camera_annotation_count        Spearman ~= -0.264
```

Interpretation:

- ego speed/translation is the strongest simple cue for final feature drift
- yaw rate also matters, though less linearly
- scene/object change proxies add meaningful signal
- denser scenes and more near objects are associated with lower consistency

Layer mean similarities correlate strongly with `similarity_L24_mean`:

```text
similarity_L8_mean  Spearman ~= 0.887, Pearson ~= 0.942
similarity_L12_mean Spearman ~= 0.904, Pearson ~= 0.948
similarity_L16_mean Spearman ~= 0.945, Pearson ~= 0.967
```

This supports an L8-probe policy: run L8 first, estimate whether deeper/full
features are needed, then decide whether to stop or continue.

## Quick Modeling Diagnostics

A quick scene-grouped cross-validation diagnostic used `GroupKFold` by
`scene_token` and predicted `similarity_L24_mean`.

Models tested:

- `Ridge` with imputation + standardization
- `HistGradientBoostingRegressor` with imputation

Results:

| feature set | model | R2 | MAE |
|---|---|---:|---:|
| metadata only | Ridge | ~0.603 +/- 0.057 | ~0.0469 |
| metadata only | HistGradientBoosting | ~0.859 +/- 0.032 | ~0.0264 |
| metadata + L8 mean | Ridge | ~0.935 +/- 0.012 | ~0.0184 |
| metadata + L8 mean | HistGradientBoosting | ~0.951 +/- 0.012 | ~0.0151 |
| metadata + L8 per camera | Ridge | ~0.935 +/- 0.012 | ~0.0184 |
| metadata + L8 per camera | HistGradientBoosting | ~0.952 +/- 0.012 | ~0.0149 |
| metadata + L8/L12/L16 | Ridge | ~0.987 +/- 0.003 | ~0.0084 |
| metadata + L8/L12/L16 | HistGradientBoosting | ~0.980 +/- 0.004 | ~0.0096 |

### Online-only metadata follow-up

A follow-up diagnostic removed metadata that would not be available online at
inference time. Excluded fields include GT/object-derived annotation counts,
object distance/size/velocity statistics, camera annotation counts/2D box
areas, and new/disappeared object proxy fields.

Using `GroupKFold` by `scene_token` and predicting `similarity_L24_mean` with
`HistGradientBoostingRegressor`:

| feature set | feature count | R2 | MAE |
|---|---:|---:|---:|
| all metadata, including offline GT-derived fields | 42 | 0.8530 +/- 0.0368 | 0.0268 +/- 0.0014 |
| online timing + ego pose/motion only | 15 | 0.7060 +/- 0.0426 | 0.0391 +/- 0.0020 |
| online timing + ego motion only, no absolute pose | 11 | 0.6784 +/- 0.0405 | 0.0414 +/- 0.0009 |

The best current online-only metadata baseline is therefore the 15-feature
timing + ego pose/motion set. It is useful, but substantially weaker than the
offline/GT-enriched metadata table.

The most valuable five online timing + ego pose/motion features by held-out
permutation importance were:

```text
delta_ego_translation_m
ego_speed_mps
abs_yaw_rate_deg_s
ego_y_m
ego_x_m
```

Using only those five features gave:

```text
R2  = 0.6813 +/- 0.0669
MAE = 0.0406 +/- 0.0028
```

For a stricter online-only set without absolute pose/location, the most
valuable five features were:

```text
delta_ego_translation_m
ego_speed_mps
abs_yaw_rate_deg_s
ego_accel_mps2
abs_yaw_accel_deg_s2
```

Using only those five features gave:

```text
R2  = 0.6823 +/- 0.0372
MAE = 0.0411 +/- 0.0012
```

Adding simple online-derived ego-history features did not improve the
metadata-only result:

| feature set | feature count | R2 | MAE |
|---|---:|---:|---:|
| raw online pose/motion | 15 | 0.7060 +/- 0.0426 | 0.0391 +/- 0.0020 |
| raw online motion only, no absolute pose | 11 | 0.6784 +/- 0.0405 | 0.0414 +/- 0.0009 |
| engineered online pose/motion history | 74 | 0.6868 +/- 0.0495 | 0.0402 +/- 0.0022 |
| engineered online motion history, no absolute pose | 65 | 0.6571 +/- 0.0525 | 0.0425 +/- 0.0009 |

This suggests that better online prediction likely requires new online signals
such as image difference, L8 feature similarity, previous detection confidence,
or predicted-box temporal change, rather than more ego-motion bookkeeping.

Main takeaway:

- metadata alone is already predictive of final representation consistency
- L8 similarity is a very strong early signal
- L12/L16 similarity is even stronger, but those features are only available
  after paying their compute cost, so they are better suited for staged/cascade
  policies than for pre-routing

## Recommended Policy Families

### 1. Metadata-only policy

This policy decides before running any encoder block.

Candidate inputs:

```text
ego_speed_mps
abs_yaw_rate_deg_s
ego_accel_mps2
abs_yaw_accel_deg_s2
time/frame_number or time_since_L24
new_object_proxy_count / fraction
near_object_count_*m
num_annotations
camera_annotation_count
avg/min object distance
```

This is the cheapest policy family and should be used as a baseline.

### 2. L8-probe policy

This policy always runs L8 first, computes L8 feature drift/similarity, then
chooses whether to stop or continue:

```text
run L8
if predicted L24 consistency is high:
    stop at L8
elif medium:
    continue to L12 or L16
else:
    run L24
```

Candidate inputs:

```text
metadata features
similarity_L8_mean
similarity_L8_<CAM> per-camera values
worst/lowest L8 camera similarity
L8 camera similarity spread
```

This is likely the most promising first learned policy because L8 is cheap and
its similarity strongly predicts L24 consistency.

### 3. Cascaded L8 -> L12 -> L16 policy

This policy uses progressively deeper evidence:

```text
run L8
if uncertain: run L12
if still uncertain: run L16
if still hard: run L24
```

This family should come after metadata-only and L8-probe baselines because it
is more complex and needs careful latency accounting.

## Candidate Labels For Stage A

Start with thresholded `similarity_L24_mean` labels:

```text
easy:   similarity_L24_mean >= 0.80
medium: 0.65 <= similarity_L24_mean < 0.80
hard:   similarity_L24_mean < 0.65
```

These thresholds should be treated as initial EDA cuts, not final policy
thresholds. The notebook should inspect class balance and adjust if needed.

Alternative regression target:

```text
y = similarity_L24_mean
```

Alternative risk target:

```text
y = 1 - similarity_L24_mean
```

Useful per-camera target variants:

```text
mean_L24_similarity
min_camera_L24_similarity
front_camera_L24_similarity
camera_spread = max_camera_L24_similarity - min_camera_L24_similarity
```

The minimum camera similarity may be more policy-relevant than the mean because
a single stale camera can hurt multi-view detection.

## Recommended Stage A: Exploratory Separability

Create a notebook or script that joins the two CSVs by `token`, filters to
`has_previous_frame=1`, and generates:

1. Target distributions:
   - `similarity_L24_mean`
   - `1 - similarity_L24_mean`
   - min per-camera L24 similarity
   - L8/L12/L16 means

2. Scatter/KDE plots:
   - speed vs L24 similarity
   - yaw rate vs L24 similarity
   - new-object proxy vs L24 similarity
   - object count / near-object count vs L24 similarity
   - L8 mean vs L24 mean
   - worst-camera L8 vs worst-camera L24

3. Label separability:
   - easy/medium/hard histograms for each candidate feature
   - per-scene class balance
   - correlation matrix for metadata and similarity features

4. Camera diagnostics:
   - per-camera L24 similarity distributions
   - per-camera L8->L24 correlations
   - worst-camera identity distribution

## Recommended Stage B: Simple Predictive Baselines

Use `GroupKFold` by `scene_token`, not random splits.

Regression baselines:

```text
metadata -> similarity_L24_mean
metadata + L8_mean -> similarity_L24_mean
metadata + L8_per_camera -> similarity_L24_mean
```

Models:

```text
Ridge
HistGradientBoostingRegressor
RandomForestRegressor only if needed for feature importances
```

Classification baselines:

```text
metadata -> easy/medium/hard
metadata + L8_mean -> easy/medium/hard
metadata + L8_per_camera -> easy/medium/hard
```

Models:

```text
LogisticRegression
HistGradientBoostingClassifier
```

Metrics:

```text
R2, MAE for regression
balanced accuracy, macro-F1, confusion matrix for classification
calibration curve if using predicted probabilities
```

## Turning Predictions Into A Policy

After Stage A/B, convert predicted similarity/risk into action thresholds:

```text
if predicted_L24_similarity >= high_threshold:
    L8
elif predicted_L24_similarity >= medium_threshold:
    L12
elif predicted_L24_similarity >= low_threshold:
    L16
else:
    L24
```

Thresholds should be selected by budget, not just accuracy. The first target is
a K=3-equivalent average compute budget.

Report for every candidate policy:

```text
action histogram
full-frame rate
average action cost
max gap between L24 calls
estimated latency budget
```

Then run actual detector evaluation only for the small number of policies that
look promising offline.

## Signals To Add Later

Inference-available signals:

- image difference from previous frame, per camera and global
- edge/gradient difference from previous frame
- cheap SSIM-like image difference
- previous-frame detection confidence, score margin, entropy
- count of previous high-confidence detections
- temporal jump of predicted boxes
- time since last L24
- previous action
- previous predicted difficulty/risk

Offline-only or GT-derived analysis signals:

- true instance enter/exit counts if instance tokens are added to the pickle
- class-specific object counts
- near-object counts by class
- object velocity/yaw statistics by class
- foreground-only feature similarity
- detector-output agreement with L24

## Immediate Recommendation

Build a small notebook or script, tentatively:

```text
tools/policy_metadata_explore.py
```

or a notebook under:

```text
notebooks/policy_metadata_explore.ipynb
```

It should:

1. load and join the metadata and similarity CSVs
2. create target variants and easy/medium/hard labels
3. generate the Stage A plots
4. run the Stage B grouped CV baselines
5. save a compact report under `outputs/policy_signal_stats/val/explore/`

The first modeling baseline to trust is:

```text
HistGradientBoostingRegressor(metadata + L8_mean -> similarity_L24_mean)
```

Reason: it is strong, cheap, and compatible with an L8-probe dynamic policy.
