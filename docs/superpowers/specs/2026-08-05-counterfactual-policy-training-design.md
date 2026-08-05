# Counterfactual Detection-Degradation Policy Training Design

## Goal

Make the planned `RepDETR3D EVA02 gated residual trainable MLP policy 4-level reset4 counterfactual detection-degradation teacher forcing no distill` result row runnable end to end. The learned policy must choose among `L8`, `L12`, `L16`, and `L24` from deployable inputs, use task degradation rather than representation similarity as its supervision target, and continue receiving counterfactual supervision after an initial updater warmup.

The main scientific hypothesis is that a policy trained to predict the shallowest depth that preserves L24 detections will allocate full computation to frames where shallow processing causes meaningful detection degradation. Reset4 remains a maximum-age safety guard, not the source of the learned L24 signal.

## Scope

The implementation will add four connected stages:

1. Convert paired L8/L12/L16/L24 prediction files into per-frame matched-prediction errors and four-class oracle actions.
2. Build deployable policy inputs from online metadata and previous-frame prediction aggregates, then pretrain and export a four-class MLP checkpoint.
3. Continue token-indexed counterfactual policy supervision during temporal updater training after a short policy-freeze warmup.
4. Add a dedicated reset4 experiment config and replace the planned RESULTS entry with complete runnable commands.

The existing prediction collector will gain an explicit `--data-split` option so targets can be collected from the training split rather than validation. It will continue to enforce `samples_per_gpu=1`, `workers_per_gpu=0`, sequential ordering, and shared pre-frame state across all four action branches.

The first implementation uses matched-prediction degradation as the required endpoint. A per-frame detector-loss CSV is an optional second constraint. Detector loss is not required because no production per-frame loss collector currently exists, and the user explicitly approved a boxes-only path when detection loss is not independently useful.

## Data flow

```text
paired branch inference (batch=1)
        |
        v
predictions_L8/L12/L16/L24 + manifest + canonical actions
        |
        v
class-aware matched-prediction error against nuScenes GT
        |
        v
shallowest action within degradation tolerance of L24
        |
        +--> counterfactual_policy_targets.csv
        |
        v
online metadata + previous canonical prediction aggregates
        |
        v
four-class MLP pretraining/export
        |
        v
reset4 updater training with loaded policy
        |
        +--> policy frozen for initial updater warmup
        +--> counterfactual CSV supervision for remaining steps
```

All target generation must use the training split. Validation targets may be generated only for policy evaluation and must never be merged into the training checkpoint.

## Counterfactual target generation

Add a reusable Python module for the matched-prediction metric and a CLI that consumes:

- `counterfactual_manifest.json`;
- `collection_progress.jsonl`;
- `predictions_L8/L12/L16/L24/results_nusc.json`;
- nuScenes ground truth;
- the policy metadata CSV for the same split;
- optionally, a per-frame detection-loss CSV.

The metric follows the existing drift-depth notebook:

- class-aware Hungarian matching;
- configurable center-distance gate;
- matched costs for confidence, center, size, yaw, and velocity;
- explicit false-positive and false-negative penalties;
- a box-only mode that omits confidence but retains geometry and unmatched-box penalties.

For action `a`, define matched-prediction degradation relative to full depth as:

```text
matched_degradation_a = error_a - error_L24
```

Negative degradation is allowed: a shallow branch that happens to score better than L24 remains eligible. Starting from L8, choose the first action satisfying:

```text
matched_degradation_a <= max_matched_degradation
```

When a detector-loss CSV is supplied, the action must also satisfy:

```text
loss_a - loss_L24 <= max_loss_degradation
```

L24 is always eligible. The initial default `max_matched_degradation` is `0.05`, exposed as a CLI argument so it can be calibrated from the generated distribution without code changes. The output records all raw errors, degradations, eligibility flags, and the selected `oracle_action`; target derivation must therefore be auditable and repeatable.

## Deployable feature construction

The target builder will join online-safe metadata by sample token. Offline-style inputs will be derived from the previous frame's canonical deployed-policy prediction, not ground truth:

- object count;
- distance statistics;
- near-object counts;
- box-dimension aggregates;
- zero-filled fields that cannot be recovered from formatted predictions.

At a scene boundary, previous-prediction fields are zero. Canonical action selection comes from `collection_progress.jsonl`; the corresponding branch prediction is used for that frame before shifting it to the next frame. This matches sequential deployment semantics and avoids current-frame oracle leakage.

The builder also shifts ground-truth metadata by one frame solely to estimate the teacher-forcing residual:

```text
residual = previous_prediction_feature - previous_GT_feature
```

Per-feature residual standard deviations are saved with the target artifact. They are training-time augmentation statistics, not policy inputs.

## Policy pretraining

Extend `tools/train_temporal_policy_mlp.py` without breaking its existing three-class similarity modes. A counterfactual mode will:

- read the prepared target/feature CSV directly;
- use labels `L8`, `L12`, `L16`, and `L24`;
- use scene-grouped cross-validation;
- report accuracy, balanced accuracy, macro-F1, confusion counts, and class frequencies;
- export a checkpoint compatible with `MlpFramePolicy`;
- store feature names, normalization statistics, label/action mappings, target provenance, thresholds, and teacher-forcing residual standard deviations.

The exported mapping is:

```text
L8  -> L8
L12 -> L12
L16 -> L16
L24 -> L24
```

Pretraining uses only online metadata and previous-prediction aggregates from the prepared CSV.

## Continued supervision during updater training

Add a counterfactual target provider behind the existing policy-supervision interface. It loads a token-to-action mapping once during model construction and validates duplicate tokens and unsupported actions immediately.

For `policy_supervision.target='counterfactual_action'`:

- derive the current sample token from `img_metas`;
- look up the oracle action and map it to the policy label index;
- compute cross-entropy from the policy logits;
- do not run a full teacher encoder solely to construct the policy target;
- fail with a clear error if a non-scene-start training sample has no target.

Add explicit supervision step bounds:

- `start_train_step`: policy receives no loss before this step and is therefore frozen;
- `max_train_steps`: final step at which supervision remains active.

The reset4 config uses an initial 10% updater-only warmup and counterfactual supervision for the remaining 90% of the 42,192-step schedule. The detector decisions remain non-differentiable argmax decisions, so detection gradients do not update the policy.

During continued supervision, teacher-forced previous-GT features receive per-feature residual noise from the loaded policy checkpoint. A scalar multiplier controls the augmentation strength; `1.0` means one measured residual standard deviation. Online-only features receive no residual noise.

## Experiment configuration

Add a new RepDETR3D config inheriting from the current online/offline teacher-forcing setup. It will:

- set `max_frames_between_full=4`;
- load the exported four-class policy checkpoint;
- retain `policy_offline_features='teacher_forcing'` so training uses previous GT and inference uses previous predictions;
- use `counterfactual_action` supervision from the training target CSV;
- freeze policy supervision for the first 10% of steps;
- supervise the policy for the remaining schedule;
- apply checkpoint residual noise with multiplier `1.0`;
- keep the pretrained detector frozen and train the existing projector/updater/policy modules;
- keep distillation disabled to isolate the policy target.

Artifact paths are config parameters and can be overridden with `--cfg-options` on another machine.

## Evaluation and policy proof

The primary validation uses reset4 and logs:

- NDS and mAP;
- per-class AP and nuScenes TP errors;
- wall-clock latency and GPU memory;
- L8/L12/L16/L24 counts;
- learned L24 count;
- forced L24 count.

Run the same checkpoint again with the age guard disabled. This second evaluation is not the primary accuracy row; it demonstrates that L24 decisions occur without the reset heuristic. A compute-matched heuristic comparison remains a separate experiment.

## Error handling and provenance

The pipeline must fail early for:

- mismatched token sets across branch prediction files;
- a manifest that does not declare shared pre-frame state;
- duplicate target tokens;
- missing metadata rows;
- unknown classes or actions;
- training targets generated from a validation artifact when the config expects the train split;
- incompatible checkpoint feature names or class counts.

The prepared target manifest records hashes of its inputs, metric parameters, split, class frequencies, thresholds, and output CSV hash. The policy checkpoint embeds the target-manifest hash.

## Tests

Add focused CPU tests for:

1. class-aware matching and unmatched penalties;
2. shallowest-eligible action selection, including negative degradation and L24 fallback;
3. optional detector-loss constraint;
4. previous canonical prediction shifting at scene boundaries;
5. four-class trainer export compatibility with `MlpFramePolicy`;
6. counterfactual token lookup and missing-token errors;
7. supervision start/end step behavior;
8. per-feature residual-noise application;
9. config loading with expected reset4, checkpoint, and target settings;
10. preservation of existing similarity-policy behavior.

Before a real run, perform a small batch-1 collection and target-generation smoke test, pretrain the MLP on the small artifact, load it through the new config, and execute a short forward/training smoke test.

## Out of scope

- Reinforcement learning or differentiable soft routing.
- Claiming the matched-prediction surrogate is equivalent to nuScenes mAP or NDS.
- Training on validation targets.
- Making detection-loss collection mandatory.
- Removing reset4 from the primary performance experiment.
