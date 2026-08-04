# Counterfactual Depth Prediction Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate token-aligned L8/L12/L16/L24 nuScenes predictions from identical per-frame temporal state and advance each sequence with the deployed policy state.

**Architecture:** Add one CLI collector with small pure helpers for cloning runtime state, temporarily forcing an action, and organizing branch execution. Reuse the repository's standard dataset, model construction, inference, and `format_results` paths. Unit tests use lightweight fake models; a GPU smoke run precedes the full validation experiment.

**Tech Stack:** Python, PyTorch, MMCV/MMDetection3D, nuScenes result formatting, pytest.

---

## File Structure

- Create: `tools/collect_depth_counterfactuals.py` — CLI, runtime-state transaction, sequential branch collection, formatting, manifest, and progress records.
- Create: `tests/test_collect_depth_counterfactuals.py` — state cloning, restoration, forced-policy cleanup, branch equality, and input validation.
- Do not modify detector runtime code or the locally executed notebook.

### Task 1: Runtime State Transaction

**Files:**
- Create: `tools/collect_depth_counterfactuals.py`
- Create: `tests/test_collect_depth_counterfactuals.py`

- [ ] **Step 1: Write failing tests for cloned state and exact restoration**

Define fake model/controller/head objects and assert:

```python
snapshot = capture_runtime_state(model)
model.temporal_memory_controller.memory[0].add_(10)
model.pts_bbox_head.memory_embedding.zero_()
model._tm_policy_prev_state['timestamp'] = 99

assert torch.equal(snapshot['controller']['memory'][0], torch.tensor([1.0]))
restore_runtime_state(model, snapshot)
assert torch.equal(model.temporal_memory_controller.memory[0], torch.tensor([1.0]))
assert torch.equal(model.pts_bbox_head.memory_embedding, torch.tensor([2.0]))
assert model._tm_policy_prev_state['timestamp'] == 3
```

Also assert a detector history attribute created after capture is deleted by restore.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_collect_depth_counterfactuals.py
```

Expected: collection error because `tools.collect_depth_counterfactuals` does not exist.

- [ ] **Step 3: Implement minimal recursive cloning and state capture**

Use:

```python
CONTROLLER_FIELDS = (
    'frame_index', 'frames_since_full', 'scene_token', 'memory',
    'last_action', 'last_action_forced', 'last_distillation_loss',
    '_policy_stats',
)
HEAD_FIELDS = (
    'memory_embedding', 'memory_reference_point', 'memory_timestamp',
    'memory_egopose', 'memory_velo', 'latest_routing_metadata',
)
MODEL_FIELDS = ('prev_scene_token', 'test_flag')

def clone_runtime_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_runtime_value(item) for item in value)
    return copy.deepcopy(value)
```

`capture_runtime_state(model)` records fixed controller/head fields plus every model attribute beginning `_tm_policy_prev_`. It also records the exact set of dynamic model history names. `restore_runtime_state` restores cloned values and deletes dynamic history attributes absent from the snapshot.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 test command.

Expected: state tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/collect_depth_counterfactuals.py tests/test_collect_depth_counterfactuals.py
git commit -m "feat: snapshot temporal inference state"
```

### Task 2: Forced Actions and Paired Branch Execution

**Files:**
- Modify: `tools/collect_depth_counterfactuals.py`
- Modify: `tests/test_collect_depth_counterfactuals.py`

- [ ] **Step 1: Write failing forced-policy tests**

Test that `forced_action(controller, 'L8')`:

- returns L24 when `has_memory=False` or `is_reset_frame=True`;
- returns L8 otherwise;
- sets `max_frames_between_full=None` only inside the context;
- restores the original policy and age limit after both success and a raised exception.

Test `collect_frame_branches(model, run_inference)` with a fake inference callback that mutates controller/head state. Assert all four callbacks observe identical pre-state, all four predictions are returned, and final live state equals the deployed-policy post-state.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
pytest -q tests/test_collect_depth_counterfactuals.py -k 'forced or branches'
```

Expected: FAIL because the context manager and branch collector are undefined.

- [ ] **Step 3: Implement the forced policy and branch transaction**

Use a policy with:

```python
class ForcedFramePolicy:
    def __init__(self, action):
        if action not in ('L8', 'L12', 'L16', 'L24'):
            raise ValueError(f'unsupported action: {action}')
        self.action = action

    def decide(self, has_memory, is_reset_frame=False, **kwargs):
        return 'L24' if is_reset_frame or not has_memory else self.action
```

The context manager saves and restores `controller.policy` and `controller.max_frames_between_full` in `finally`.

`collect_frame_branches` performs:

```python
pre_state = capture_runtime_state(model)
canonical_prediction = run_inference(None)
canonical_action = model.temporal_memory_controller.last_action
canonical_post_state = capture_runtime_state(model)
predictions = {canonical_action: canonical_prediction}
for action in ACTIONS:
    if action == canonical_action:
        continue
    restore_runtime_state(model, pre_state)
    with forced_action(model.temporal_memory_controller, action):
        predictions[action] = run_inference(action)
restore_runtime_state(model, canonical_post_state)
return predictions, canonical_action
```

- [ ] **Step 4: Run all collector tests and verify GREEN**

```bash
pytest -q tests/test_collect_depth_counterfactuals.py
```

Expected: all state and branch tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/collect_depth_counterfactuals.py tests/test_collect_depth_counterfactuals.py
git commit -m "feat: collect paired depth branches"
```

### Task 3: CLI, Sequential Inference, and Official Results

**Files:**
- Modify: `tools/collect_depth_counterfactuals.py`
- Modify: `tests/test_collect_depth_counterfactuals.py`

- [ ] **Step 1: Write failing validation and manifest tests**

Cover:

```python
validate_output_dir(nonempty_path)  # raises FileExistsError
validate_model(fake_without_temporal_controller)  # raises ValueError
validate_token_sets({'L8': ['a'], 'L12': ['b'], 'L16': ['a'], 'L24': ['a']})  # raises ValueError
```

Test that `build_manifest` includes shared-state truth, deployed-policy advancement, four paths, action counts, sample count, and a deterministic SHA-256 hash of the sorted token set.

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest -q tests/test_collect_depth_counterfactuals.py -k 'validate or manifest'
```

Expected: FAIL because validation/manifest helpers are undefined.

- [ ] **Step 3: Add CLI and repository-native setup**

Arguments:

```text
config checkpoint --output-dir --num-samples --device --cfg-options
```

Inside `main`, lazily import MMCV/MMDetection3D components, load the config plugin, force test batch size and workers to one/zero, preserve sequential ordering, build the model, load the checkpoint, wrap it with `MMDataParallel`, and validate temporal modules and all three layered projectors.

For each dataloader item, call the parallel model through a closure, normalize the returned single prediction item, collect four branch lists, append one JSONL progress record, and print `[current/total] token canonical=<action>`.

- [ ] **Step 4: Format and copy official result JSON files**

For each action, call `dataset.format_results`, resolve its `pts_bbox/results_nusc.json`, and copy it to:

```text
OUTPUT_DIR/predictions_<ACTION>/results_nusc.json
```

Validate identical token sets from the JSON `results` mappings. Write `counterfactual_manifest.json` and `collection_summary.json` only after all four files validate.

- [ ] **Step 5: Run all unit tests**

```bash
pytest -q tests/test_collect_depth_counterfactuals.py
```

Expected: all tests pass without loading a detector checkpoint.

- [ ] **Step 6: Commit**

```bash
git add tools/collect_depth_counterfactuals.py tests/test_collect_depth_counterfactuals.py
git commit -m "feat: export counterfactual depth predictions"
```

### Task 4: GPU Smoke Experiment and Scientific Checks

**Files:**
- Generate: `outputs/drift_depth_analysis_smoke/`

- [ ] **Step 1: Confirm the exact command with the user**

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> python tools/collect_depth_counterfactuals.py \
  projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_online_offline_teacher_forcing_no_distill.py \
  work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_no_distill/iter_42192.pth \
  --output-dir outputs/drift_depth_analysis_smoke \
  --num-samples 20 \
  --device cuda:0 \
  --cfg-options model.temporal_memory.max_frames_between_full=4
```

- [ ] **Step 2: Run and monitor the confirmed smoke command**

Monitor process liveness, `collection_progress.jsonl`, GPU memory, and a 30-minute hard timeout. Do not retry automatically.

- [ ] **Step 3: Validate smoke artifacts**

Assert four non-empty prediction files, equal token sets, manifest shared-state truth, and at least one non-start token whose shallow prediction JSON differs from L24. Load the manifest with the notebook's `validate_manifest` function.

- [ ] **Step 4: Run relevant regression tests**

```bash
pytest -q tests/test_collect_depth_counterfactuals.py tests/test_temporal_memory_controller.py tests/test_temporal_memory_repdetr3d_routing.py
```

Expected: all tests pass.

### Task 5: Full Validation Collection and Figure Generation

**Files:**
- Generate: `outputs/drift_depth_analysis/`
- Generate: `paper/figures/fig_drift_vs_depth_degradation.pdf`
- Generate: `paper/figures/fig_drift_vs_depth_degradation_scatter.pdf`

- [ ] **Step 1: Confirm the full command with the user**

Use the smoke command without `--num-samples` and with `--output-dir outputs/drift_depth_analysis`.

- [ ] **Step 2: Run and monitor full collection**

Monitor process liveness and progress growth every 30-60 seconds. Use a four-hour hard timeout and never auto-retry a crash.

- [ ] **Step 3: Validate full artifacts**

Require identical token counts and hashes across four prediction files, 5,869 joinable non-scene-start drift rows, non-empty summary/manifest files, and canonical action counts summing to the number of collected frames.

- [ ] **Step 4: Execute the production notebook**

Run all notebook code cells with `MPLBACKEND=Agg` and synthetic mode unset. Confirm the two paper PDFs, per-frame degradation CSV, quantile summary, correlation summary, and validation summary are non-empty.

- [ ] **Step 5: Report experiment result**

Record exact command, checkpoint/config hashes, exit status, elapsed time, output paths and sizes, token counts, canonical action distribution, anomalies, and whether figure generation completed.
