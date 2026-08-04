# Counterfactual Depth Prediction Collector

## Objective

Generate aligned nuScenes validation predictions for L8, L12, L16, and L24 so the drift-versus-depth-degradation notebook can test whether full-representation drift identifies frames that require deeper computation.

The four predictions for a frame must differ only in the current-frame encoder-depth action. Dense temporal memory, detector query memory, policy history, scene position, input data, and model parameters must be identical at the start of every branch.

## Model and Checkpoint

Use:

- config: `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_online_offline_teacher_forcing_no_distill.py`;
- checkpoint: `work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_no_distill/iter_42192.pth`;
- config override: `model.temporal_memory.max_frames_between_full=4`.

The checkpoint's 545 image-backbone tensors and 283 detector-head tensors are bitwise identical to the base checkpoint. Its L24 branch is therefore the base detector path, while its 113 temporal-only tensors provide trained L8/L12/L16 projectors and memory updating.

Use the existing base-checkpoint table `outputs/policy_signal_stats/val/policy_feature_similarity.csv` for adjacent-frame L24 drift. Because the image backbone is identical, this drift is aligned with the collector's L24 representation.

## Counterfactual Protocol

For every validation frame:

1. Snapshot every mutable inference state.
2. Evaluate the deployed policy once, recording its selected action, prediction, and post-frame state.
3. Restore the pre-frame snapshot before every remaining depth branch.
4. Evaluate the other three actions under a temporary forced policy.
5. Restore the saved deployed-policy post-frame state to advance the sequence.
6. Store all four branch predictions under the same sample token.

On scene-start frames, missing dense memory legitimately forces L24 behavior. The collector still emits all four aligned records, and the downstream notebook excludes those frames because adjacent-frame drift is undefined.

The canonical deployed-policy trajectory is preferable to an all-L24 trajectory because it measures depth sensitivity under the paper's actual temporal-memory operating regime. The current-frame branch comparison remains paired because each branch begins from the same pre-frame state.

## Mutable State Contract

Snapshot and restore tensor values by cloning rather than retaining aliases.

### Temporal memory controller

- `frame_index`;
- `frames_since_full`;
- `scene_token`;
- dense `memory` feature list;
- `last_action` and `last_action_forced`;
- policy statistics;
- any last distillation-loss field.

### Sparse detector head

- `memory_embedding`;
- `memory_reference_point`;
- `memory_timestamp`;
- `memory_egopose`;
- `memory_velo`;
- latest routing metadata, when present.

### Detector-level inference history

- `prev_scene_token`;
- previous prediction-derived policy features;
- previous L8/L12 policy probe features;
- any other `_tm_policy_prev*` field present on the model.

The collector must test snapshot independence by modifying live tensor state after a snapshot and confirming the snapshot did not change.

## Forcing a Branch

Temporarily replace only the controller's policy decision with a constant action in `{L8, L12, L16, L24}`. Preserve all trained modules and configuration. Temporarily disable only the maximum-memory-age override while a branch runs so it cannot replace a requested shallow action with L24. Restore the original policy and age limit after each branch, including exceptions.

Obtain the canonical action by executing one normal deployed-policy pass from the pre-frame state. Preserve that pass's post-state, restore the pre-state for the other branches, and finally restore the canonical post-state. This avoids an additional policy-probe encoder pass while producing the same continuing trajectory as ordinary inference. Do not derive the canonical action from ground truth.

## Collector Interface

Add one small CLI tool:

```text
python tools/collect_depth_counterfactuals.py \
  CONFIG CHECKPOINT \
  --output-dir outputs/drift_depth_analysis \
  --cfg-options model.temporal_memory.max_frames_between_full=4
```

Support:

- `--num-samples` for a short smoke run;
- `--device` with one GPU per process;
- `--score-threshold` only for optional output-size reduction, defaulting to no extra filtering;
- `--cfg-options` following existing repository conventions.

Batch size must be one and validation order must remain sequential. The collector stops if the model is not RepDETR3D temporal memory with trained L8/L12/L16 projectors.

## Outputs

Write:

```text
outputs/drift_depth_analysis/
  predictions_L8/results_nusc.json
  predictions_L12/results_nusc.json
  predictions_L16/results_nusc.json
  predictions_L24/results_nusc.json
  counterfactual_manifest.json
  collection_progress.jsonl
  collection_summary.json
```

Use the dataset's normal `format_results` path so each prediction file follows the official nuScenes schema.

The manifest records:

- `shared_pre_frame_state: true`;
- run ID and timestamp;
- config and checkpoint absolute paths;
- checkpoint hashes;
- canonical state-advance mode: `deployed_policy`;
- prediction paths;
- sample-token count and exact token-set hash;
- per-action counts for the canonical trajectory;
- state fields covered by snapshot/restore;
- relevant config overrides.

Write progress after each completed frame so a crash is diagnosable. Do not silently resume a partial run; require a new output directory or an explicit future resume feature.

## Validation

### Unit tests

- snapshot clones tensors and nested containers without copying model parameters;
- restore reproduces controller, head, and detector history exactly;
- a forced branch always restores policy and age settings, including on exceptions;
- restoring the saved canonical post-state produces the same next-frame state as ordinary single-pass inference;
- output token sets are identical across all four depths;
- invalid batch size, missing temporal modules, and non-empty output directories fail clearly.

### Smoke experiment

Run two validation scenes or at least 20 sequential frames on one GPU. Confirm:

- all four prediction files are non-empty and token-aligned;
- scene starts produce identical forced-full results;
- at least one non-start frame produces distinct shallow and L24 predictions;
- the canonical post-state matches a normal single-pass canonical inference on a short duplicate sequence within numerical tolerance;
- manifest validation passes in `notebooks/drift_quantile_vs_depth_degradation.ipynb`.

### Full experiment

Run the complete validation split only after the smoke experiment passes. Monitor process liveness, progress-file growth, GPU memory, and hard timeout. Do not auto-retry a crash.

After collection, execute the drift-depth notebook in production mode and verify that both paper PDFs and all statistics artifacts are non-empty.

## Non-goals

- Retraining or changing the selected checkpoint.
- Mutating detector behavior for normal evaluation.
- Computing four independent temporal rollouts.
- Claiming per-frame matched-prediction error is mAP or NDS.
- Implementing distributed multi-GPU collection before a measured need.
