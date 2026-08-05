# Counterfactual Policy Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable four-depth policy pipeline supervised by per-frame detection degradation relative to L24.

**Architecture:** A batch-1 branch collector produces paired predictions. A CPU target builder computes class-aware matched-prediction error, assigns the shallowest acceptable depth, and builds deployable previous-prediction features. The existing MLP trainer exports a four-class checkpoint; a token target provider supplies continued cross-entropy supervision to RepDETR3D after an updater-only warmup.

**Tech Stack:** Python 3.8, PyTorch, pandas, NumPy, SciPy Hungarian matching, MMDetection3D/MMCV configs, pytest.

---

## File structure

- Create `tools/counterfactual_policy_targets.py`: metric, oracle-action selection, previous-prediction feature preparation, CSV/manifest CLI.
- Create `tests/test_counterfactual_policy_targets.py`: CPU unit tests for matching, action selection, shifting, and validation.
- Modify `tools/collect_depth_counterfactuals.py`: explicit train/val/test split selection while retaining batch size 1.
- Modify `tests/test_collect_depth_counterfactuals.py`: split-selection tests.
- Modify `tools/train_temporal_policy_mlp.py`: backward-compatible four-class counterfactual mode and checkpoint provenance.
- Create `tests/test_train_temporal_policy_mlp.py`: export/load compatibility tests.
- Create `projects/mmdet3d_plugin/models/temporal_memory/policy_targets.py`: token-indexed action provider.
- Modify `projects/mmdet3d_plugin/models/temporal_memory/policies.py`: load residual-noise statistics from exported checkpoints.
- Modify `projects/mmdet3d_plugin/models/detectors/repdetr3d.py`: CSV target loss, start/end supervision bounds, measured residual noise, and no unnecessary full teacher.
- Modify `tests/test_temporal_memory_repdetr3d_routing.py`: supervision behavior tests.
- Create `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_4level_reset4_counterfactual_teacher_forcing_no_distill.py`: runnable experiment.
- Modify `RESULTS.md`: replace the planned note with collection, target, pretraining, training, and evaluation commands.

### Task 1: Matched-prediction metric and oracle action

**Files:**
- Create: `tools/counterfactual_policy_targets.py`
- Create: `tests/test_counterfactual_policy_targets.py`

- [ ] **Step 1: Write failing metric and action-selection tests**

```python
def test_matched_error_is_class_aware_and_penalizes_unmatched():
    predictions = [box('car', x=0), box('pedestrian', x=20)]
    targets = [box('car', x=1)]
    error = matched_prediction_error(predictions, targets, MetricConfig())
    assert error > 0
    assert error == pytest.approx(
        matched_prediction_error([predictions[0]], targets, MetricConfig())
        + MetricConfig().false_positive_penalty)


def test_choose_shallowest_eligible_action():
    errors = {'L8': 1.04, 'L12': 1.02, 'L16': 1.01, 'L24': 1.00}
    assert choose_oracle_action(errors, max_matched_degradation=0.03) == 'L12'


def test_negative_degradation_keeps_shallow_action_eligible():
    errors = {'L8': 0.90, 'L12': 0.95, 'L16': 0.98, 'L24': 1.00}
    assert choose_oracle_action(errors, max_matched_degradation=0.0) == 'L8'


def test_optional_loss_constraint_can_reject_shallow_action():
    errors = {'L8': 1.01, 'L12': 1.01, 'L16': 1.01, 'L24': 1.00}
    losses = {'L8': 1.20, 'L12': 1.04, 'L16': 1.01, 'L24': 1.00}
    assert choose_oracle_action(errors, 0.05, losses, 0.05) == 'L12'
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `pytest -q tests/test_counterfactual_policy_targets.py`

Expected: FAIL because `tools.counterfactual_policy_targets` does not exist.

- [ ] **Step 3: Implement the metric and action selector**

```python
ACTIONS = ('L8', 'L12', 'L16', 'L24')

@dataclass(frozen=True)
class MetricConfig:
    center_gate_m: float = 4.0
    confidence_weight: float = 0.25
    center_weight: float = 1.0
    size_weight: float = 0.25
    yaw_weight: float = 0.25
    velocity_weight: float = 0.25
    false_positive_penalty: float = 1.0
    false_negative_penalty: float = 1.0
    box_only: bool = False

def matched_prediction_error(predictions, targets, config):
    total = 0.0
    matched_predictions, matched_targets = set(), set()
    classes = sorted({item['class_name'] for item in predictions + targets})
    for class_name in classes:
        pred_ids = [i for i, item in enumerate(predictions)
                    if item['class_name'] == class_name]
        target_ids = [i for i, item in enumerate(targets)
                      if item['class_name'] == class_name]
        if not pred_ids or not target_ids:
            continue
        matrix = np.full((len(pred_ids), len(target_ids)), np.inf)
        for row, pred_id in enumerate(pred_ids):
            for column, target_id in enumerate(target_ids):
                cost = pair_cost(predictions[pred_id], targets[target_id], config)
                if cost is not None:
                    matrix[row, column] = cost
        finite = np.isfinite(matrix)
        if not finite.any():
            continue
        fill = matrix[finite].max() + config.false_positive_penalty + config.false_negative_penalty
        rows, columns = linear_sum_assignment(np.where(finite, matrix, fill))
        for row, column in zip(rows, columns):
            if finite[row, column]:
                total += matrix[row, column]
                matched_predictions.add(pred_ids[row])
                matched_targets.add(target_ids[column])
    total += config.false_positive_penalty * (len(predictions) - len(matched_predictions))
    total += config.false_negative_penalty * (len(targets) - len(matched_targets))
    return total / max(1, len(predictions) + len(targets))

def choose_oracle_action(errors, max_matched_degradation,
                         losses=None, max_loss_degradation=None):
    for action in ACTIONS:
        matched_ok = errors[action] - errors['L24'] <= max_matched_degradation
        loss_ok = losses is None or (
            losses[action] - losses['L24'] <= max_loss_degradation)
        if matched_ok and loss_ok:
            return action
    return 'L24'
```

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_counterfactual_policy_targets.py -k 'matched or shallowest or loss_constraint'`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/counterfactual_policy_targets.py tests/test_counterfactual_policy_targets.py
git commit -m "feat: add counterfactual depth target metric"
```

### Task 2: Build auditable target and deployable-feature artifacts

**Files:**
- Modify: `tools/counterfactual_policy_targets.py`
- Modify: `tests/test_counterfactual_policy_targets.py`

- [ ] **Step 1: Add failing tests for canonical prediction shifting and validation**

```python
def test_previous_prediction_features_follow_canonical_action_and_reset_at_scene():
    frames = [
        {'token': 'a0', 'scene_token': 'A', 'canonical_action': 'L8'},
        {'token': 'a1', 'scene_token': 'A', 'canonical_action': 'L24'},
        {'token': 'b0', 'scene_token': 'B', 'canonical_action': 'L12'},
    ]
    rows = build_previous_prediction_rows(frames, branch_results())
    assert rows[0]['num_annotations'] == 0
    assert rows[1]['num_annotations'] == 1  # a0's L8 prediction
    assert rows[2]['num_annotations'] == 0


def test_manifest_requires_shared_pre_frame_state(tmp_path):
    manifest = {'shared_pre_frame_state': False}
    with pytest.raises(ValueError, match='shared pre-frame state'):
        validate_manifest(manifest)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `pytest -q tests/test_counterfactual_policy_targets.py -k 'previous_prediction or manifest'`

Expected: FAIL because the builder functions are absent.

- [ ] **Step 3: Implement artifact construction and CLI**

The CLI must accept the following exact interface:

```python
parser.add_argument('--counterfactual-dir', required=True)
parser.add_argument('--metadata-csv', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--data-root', default='data/nuscenes')
parser.add_argument('--version', default='v1.0-trainval')
parser.add_argument('--split', required=True, choices=('train', 'val'))
parser.add_argument('--max-matched-degradation', type=float, default=0.05)
parser.add_argument('--detection-loss-csv')
parser.add_argument('--max-loss-degradation', type=float)
parser.add_argument('--box-only', action='store_true')
```

Implement:

```python
def prediction_features(detections, score_threshold=0.1):
    boxes = [item for item in detections
             if float(item.get('detection_score', 0.0)) >= score_threshold]
    features = zero_offline_features()
    if not boxes:
        return features
    xy = np.asarray([item['translation'][:2] for item in boxes], dtype=np.float64)
    distances = np.linalg.norm(xy, axis=1)
    sizes = np.asarray([item['size'] for item in boxes], dtype=np.float64)
    features.update(num_annotations=float(len(boxes)),
                    avg_object_distance_m=float(distances.mean()),
                    median_object_distance_m=float(np.median(distances)),
                    min_object_distance_m=float(distances.min()),
                    p10_object_distance_m=float(np.quantile(distances, 0.1)),
                    p90_object_distance_m=float(np.quantile(distances, 0.9)),
                    avg_box_length_m=float(sizes[:, 0].mean()),
                    avg_box_width_m=float(sizes[:, 1].mean()),
                    avg_box_height_m=float(sizes[:, 2].mean()),
                    avg_box_volume_m3=float(np.prod(sizes, axis=1).mean()),
                    avg_box_area_m2=float(np.prod(sizes[:, :2], axis=1).mean()))
    for radius in (10, 20, 30, 50):
        features[f'near_object_count_{radius}m'] = float((distances < radius).sum())
    return features

def build_previous_prediction_rows(frames, branch_results):
    rows, previous_scene, previous_features = [], None, None
    for frame in frames:
        scene = frame['scene_token']
        rows.append(zero_offline_features() if scene != previous_scene
                    else dict(previous_features))
        action = frame['canonical_action']
        previous_features = prediction_features(branch_results[action][frame['token']])
        previous_scene = scene
    return rows

def build_target_rows(manifest, progress, branch_results, ground_truth,
                      metadata, metric_config, matched_threshold,
                      loss_rows=None, loss_threshold=None):
    previous_rows = build_previous_prediction_rows(progress, branch_results)
    output = []
    for frame, previous in zip(progress, previous_rows):
        token = frame['token']
        errors = {action: matched_prediction_error(
            branch_results[action][token], ground_truth[token], metric_config)
            for action in ACTIONS}
        losses = None if loss_rows is None else loss_rows[token]
        row = dict(metadata[token])
        row.update(previous)
        row.update({f'error_{action}': errors[action] for action in ACTIONS})
        row.update({f'degradation_{action}': errors[action] - errors['L24']
                    for action in ACTIONS})
        row['oracle_action'] = choose_oracle_action(
            errors, matched_threshold, losses, loss_threshold)
        output.append(row)
    return output
```

Write `counterfactual_policy_targets.csv` and `counterfactual_policy_targets_manifest.json`. The manifest includes source hashes, split, metric parameters, thresholds, class counts, residual standard deviations, and output CSV hash.

- [ ] **Step 4: Run all target-builder tests**

Run: `pytest -q tests/test_counterfactual_policy_targets.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/counterfactual_policy_targets.py tests/test_counterfactual_policy_targets.py
git commit -m "feat: build counterfactual policy targets"
```

### Task 3: Support train-split batch-1 collection

**Files:**
- Modify: `tools/collect_depth_counterfactuals.py`
- Modify: `tests/test_collect_depth_counterfactuals.py`

- [ ] **Step 1: Write failing split-selection tests**

```python
def test_select_dataset_cfg_uses_requested_split():
    cfg = SimpleNamespace(data={'train': {'ann_file': 'train.pkl'},
                                'val': {'ann_file': 'val.pkl'},
                                'test': {'ann_file': 'test.pkl'}})
    selected = select_dataset_cfg(cfg, 'train')
    assert selected['ann_file'] == 'train.pkl'
    assert selected['test_mode'] is True
    assert 'samples_per_gpu' not in selected


def test_parse_args_accepts_train_split(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['tool', 'cfg.py', 'model.pth',
                                     '--data-split', 'train'])
    assert parse_args().data_split == 'train'
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest -q tests/test_collect_depth_counterfactuals.py -k 'split'`

Expected: FAIL because `--data-split` and `select_dataset_cfg` do not exist.

- [ ] **Step 3: Implement split selection**

```python
parser.add_argument('--data-split', default='val',
                    choices=('train', 'val', 'test'))

def select_dataset_cfg(cfg, split):
    return prepare_test_dataset_cfg(cfg.data[split])
```

Use `select_dataset_cfg(cfg, args.data_split)` in `main`. Preserve `samples_per_gpu=1`, `workers_per_gpu=0`, `dist=False`, and `shuffle=False`. Record `data_split` in the manifest.

- [ ] **Step 4: Run collector tests**

Run: `pytest -q tests/test_collect_depth_counterfactuals.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/collect_depth_counterfactuals.py tests/test_collect_depth_counterfactuals.py
git commit -m "feat: collect train-split depth counterfactuals"
```

### Task 4: Export a compatible four-class policy checkpoint

**Files:**
- Modify: `tools/train_temporal_policy_mlp.py`
- Modify: `projects/mmdet3d_plugin/models/temporal_memory/policies.py`
- Create: `tests/test_train_temporal_policy_mlp.py`

- [ ] **Step 1: Write failing counterfactual trainer tests**

```python
def test_counterfactual_mode_uses_four_actions(tmp_path):
    frame = synthetic_counterfactual_frame()
    features, labels, actions = prepare_counterfactual_training(frame)
    assert labels == ['L8', 'L12', 'L16', 'L24']
    assert actions == {name: name for name in labels}
    assert features == EXPECTED_DEPLOYABLE_FEATURES


def test_export_loads_in_mlp_frame_policy(tmp_path):
    path = export_synthetic_policy(tmp_path)
    policy = MlpFramePolicy(checkpoint=str(path))
    assert policy.labels == ['L8', 'L12', 'L16', 'L24']
    assert len(policy.teacher_forcing_residual_std) == len(policy.feature_names)
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest -q tests/test_train_temporal_policy_mlp.py`

Expected: FAIL because counterfactual mode and residual checkpoint fields are absent.

- [ ] **Step 3: Generalize the trainer without changing legacy defaults**

Add:

```python
parser.add_argument('--counterfactual-csv')
parser.add_argument('--target-manifest')
```

When `--counterfactual-csv` is set:

```python
labels = ['L8', 'L12', 'L16', 'L24']
actions = {label: label for label in labels}
df = pd.read_csv(args.counterfactual_csv)
features = deployable_feature_columns(df)
y = df['oracle_action'].map({label: i for i, label in enumerate(labels)}).to_numpy()
```

Construct `PolicyMlp(in_dim=x.shape[1], hidden_dim=args.hidden_dim, out_dim=len(labels))` and parameterize `fit_model` and `evaluate_cv` by class count. Export `teacher_forcing_residual_std`, `target_manifest_sha256`, `class_counts`, and the existing normalization fields. Keep current three-class similarity behavior unchanged when `--counterfactual-csv` is absent.

- [ ] **Step 4: Run trainer and policy tests**

Run: `pytest -q tests/test_train_temporal_policy_mlp.py tests/test_temporal_memory_policies.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/train_temporal_policy_mlp.py tests/test_train_temporal_policy_mlp.py projects/mmdet3d_plugin/models/temporal_memory/policies.py
git commit -m "feat: export four-class counterfactual policy"
```

### Task 5: Token-indexed targets and measured residual noise

**Files:**
- Create: `projects/mmdet3d_plugin/models/temporal_memory/policy_targets.py`
- Modify: `projects/mmdet3d_plugin/models/temporal_memory/policies.py`
- Modify: `tests/test_temporal_memory_repdetr3d_routing.py`

- [ ] **Step 1: Write failing provider and residual-noise tests**

```python
def test_counterfactual_target_provider_rejects_duplicates(tmp_path):
    path = write_targets(tmp_path, [('token', 'L8'), ('token', 'L24')])
    with pytest.raises(ValueError, match='duplicate'):
        CounterfactualActionTargets(path)


def test_counterfactual_target_provider_returns_label_index(tmp_path):
    path = write_targets(tmp_path, [('a', 'L16')])
    provider = CounterfactualActionTargets(path)
    assert provider.label_index('a', ['L8', 'L12', 'L16', 'L24']) == 2


def test_residual_noise_only_changes_offline_features():
    policy = fake_policy(features=['frame_number', 'num_annotations'],
                         residual_std=[0.0, 2.0])
    noisy = detector._temporal_memory_noisy_policy_features(
        {'frame_number': 4.0, 'num_annotations': 10.0})
    assert noisy['frame_number'] == 4.0
    assert noisy['num_annotations'] != 10.0
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest -q tests/test_temporal_memory_repdetr3d_routing.py -k 'counterfactual_target or residual_noise'`

Expected: FAIL because the provider and residual buffer are absent.

- [ ] **Step 3: Implement provider and policy buffer**

```python
class CounterfactualActionTargets:
    ACTIONS = ('L8', 'L12', 'L16', 'L24')

    def __init__(self, csv_path, token_column='token', action_column='oracle_action'):
        frame = pd.read_csv(csv_path, usecols=[token_column, action_column])
        if frame[token_column].duplicated().any():
            raise ValueError('duplicate counterfactual policy target token')
        unknown = set(frame[action_column]) - set(self.ACTIONS)
        if unknown:
            raise ValueError(f'unsupported counterfactual actions: {sorted(unknown)}')
        self.targets = dict(zip(frame[token_column].astype(str), frame[action_column]))

    def label_index(self, token, labels):
        action = self.targets[token]
        return list(labels).index(action)
```

In `MlpFramePolicy`, register a `teacher_forcing_residual_std` buffer loaded from the checkpoint, defaulting to zeros for legacy checkpoints.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_temporal_memory_repdetr3d_routing.py -k 'counterfactual_target or residual_noise'`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/mmdet3d_plugin/models/temporal_memory/policy_targets.py projects/mmdet3d_plugin/models/temporal_memory/policies.py tests/test_temporal_memory_repdetr3d_routing.py
git commit -m "feat: add token-indexed policy targets"
```

### Task 6: Integrate continued supervision into RepDETR3D

**Files:**
- Modify: `projects/mmdet3d_plugin/models/detectors/repdetr3d.py`
- Modify: `tests/test_temporal_memory_repdetr3d_routing.py`

- [ ] **Step 1: Write failing schedule and no-teacher tests**

```python
def test_counterfactual_supervision_starts_after_warmup():
    detector.temporal_memory['policy_supervision'].update(
        target='counterfactual_action', start_train_step=10, max_train_steps=100)
    detector._tm_train_step = 9
    assert not detector._temporal_memory_policy_supervision_active()
    detector._tm_train_step = 10
    assert detector._temporal_memory_policy_supervision_active()


def test_counterfactual_target_does_not_require_full_teacher():
    detector.temporal_memory['policy_supervision']['target'] = 'counterfactual_action'
    assert not detector._temporal_memory_policy_supervision_requires_teacher()


def test_missing_non_scene_start_target_fails():
    with pytest.raises(KeyError, match='counterfactual policy target'):
        detector._temporal_memory_policy_target_index_from_token('missing')
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest -q tests/test_temporal_memory_repdetr3d_routing.py -k 'supervision_starts or require_full_teacher or missing_non_scene'`

Expected: FAIL because step bounds and CSV targets are not integrated.

- [ ] **Step 3: Implement target-aware supervision**

Add during detector initialization:

```python
if policy_spec.get('target') == 'counterfactual_action':
    self._tm_counterfactual_targets = CounterfactualActionTargets(
        policy_spec['target_csv'],
        policy_spec.get('token_column', 'token'),
        policy_spec.get('action_column', 'oracle_action'))
```

Add:

```python
def _temporal_memory_policy_supervision_requires_teacher(self):
    spec = self.temporal_memory.get('policy_supervision', {}) or {}
    return spec.get('target', 'similarity') not in ('counterfactual_action',)

def _temporal_memory_policy_supervision_active(self):
    spec = self.temporal_memory.get('policy_supervision', None)
    if not spec or spec.get('enabled', True) is False:
        return False
    step = getattr(self, '_tm_train_step', 0)
    start = int(spec.get('start_train_step', 0) or 0)
    maximum = int(spec.get('max_train_steps', 0) or 0)
    return step >= start and (maximum <= 0 or step < maximum)
```

Pass the current token into `_temporal_memory_policy_supervision_loss`. For `counterfactual_action`, get the target from the provider and do not require `previous_full/current_full`. Gate the full policy-teacher feature extraction with `_temporal_memory_policy_supervision_requires_teacher()`.

For residual noise, apply `torch.randn_like(residual_std) * residual_std * residual_noise_multiplier` only where residual std is nonzero.

- [ ] **Step 4: Run RepDETR3D temporal tests**

Run: `pytest -q tests/test_temporal_memory_repdetr3d_routing.py tests/test_temporal_memory_controller.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/mmdet3d_plugin/models/detectors/repdetr3d.py tests/test_temporal_memory_repdetr3d_routing.py
git commit -m "feat: supervise policy with counterfactual actions"
```

### Task 7: Add runnable config, commands, and smoke verification

**Files:**
- Create: `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_4level_reset4_counterfactual_teacher_forcing_no_distill.py`
- Modify: `RESULTS.md`
- Test: `tests/test_temporal_memory_configs.py`

- [ ] **Step 1: Write a failing config test**

```python
def test_counterfactual_reset4_config():
    cfg = Config.fromfile(COUNTERFACTUAL_CONFIG)
    tm = cfg.model.temporal_memory
    assert tm.max_frames_between_full == 4
    assert tm.policy.load_weights is True
    assert tm.policy_supervision.target == 'counterfactual_action'
    assert tm.policy_supervision.start_train_step == 4219
    assert tm.policy_supervision.max_train_steps == 42192
    assert tm.policy_offline_features == 'teacher_forcing'
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest -q tests/test_temporal_memory_configs.py -k counterfactual_reset4`

Expected: FAIL because the config does not exist.

- [ ] **Step 3: Add the config**

```python
_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_online_offline_teacher_forcing_no_distill.py']

policy_checkpoint = 'outputs/counterfactual_policy_train/mlp_policy_counterfactual.pt'
target_csv = 'outputs/counterfactual_policy_train/counterfactual_policy_targets.csv'

model = dict(temporal_memory=dict(
    max_frames_between_full=4,
    policy=dict(checkpoint=policy_checkpoint, load_weights=True),
    policy_supervision=dict(
        enabled=True,
        target='counterfactual_action',
        target_csv=target_csv,
        start_train_step=4219,
        max_train_steps=42192,
        residual_noise_multiplier=1.0,
        class_weights=(1.0, 1.0, 1.0, 3.0),
        weight=1.0,
    ),
    policy_offline_features='teacher_forcing',
))
```

Set a unique `work_dir` and retain no-distillation settings from the base config.

- [ ] **Step 4: Replace the planned RESULTS command**

Document exact commands for:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/collect_depth_counterfactuals.py projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_online_offline_teacher_forcing_no_distill.py work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_feature_noise_no_distill/iter_42192.pth --data-split train --output-dir outputs/counterfactual_policy_train/branches --device cuda:0 --cfg-options model.temporal_memory.max_frames_between_full=4
python tools/counterfactual_policy_targets.py --counterfactual-dir outputs/counterfactual_policy_train/branches --metadata-csv outputs/policy_signal_stats/train/policy_signal_stats.csv --output-dir outputs/counterfactual_policy_train --split train --max-matched-degradation 0.05
python tools/train_temporal_policy_mlp.py --variant offline --counterfactual-csv outputs/counterfactual_policy_train/counterfactual_policy_targets.csv --target-manifest outputs/counterfactual_policy_train/counterfactual_policy_targets_manifest.json --out outputs/counterfactual_policy_train/mlp_policy_counterfactual.pt
bash tools/dist_train.sh projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_4level_reset4_counterfactual_teacher_forcing_no_distill.py 2
```

Also include reset4 and guard-disabled evaluation commands with `model.tm_debug_latency=True`.

- [ ] **Step 5: Run CPU verification**

Run:

```bash
pytest -q \
  tests/test_counterfactual_policy_targets.py \
  tests/test_collect_depth_counterfactuals.py \
  tests/test_train_temporal_policy_mlp.py \
  tests/test_temporal_memory_policies.py \
  tests/test_temporal_memory_repdetr3d_routing.py \
  tests/test_temporal_memory_controller.py \
  tests/test_temporal_memory_configs.py
```

Expected: all tests PASS.

- [ ] **Step 6: Run lightweight artifact smoke test**

Use synthetic fixtures to run target creation, four-class policy export, and config checkpoint loading without a long detector job. Then, if a GPU and nuScenes data are available, collect `--num-samples 2` with batch size 1 and verify four branch files share the same two tokens.

- [ ] **Step 7: Commit**

```bash
git add projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_4level_reset4_counterfactual_teacher_forcing_no_distill.py RESULTS.md tests/test_temporal_memory_configs.py
git commit -m "feat: add counterfactual policy experiment"
```

### Task 8: Final regression and handoff

**Files:**
- Verify all files above.

- [ ] **Step 1: Run diff and syntax checks**

Run:

```bash
git diff --check
python -m py_compile \
  tools/counterfactual_policy_targets.py \
  tools/collect_depth_counterfactuals.py \
  tools/train_temporal_policy_mlp.py \
  projects/mmdet3d_plugin/models/temporal_memory/policy_targets.py \
  projects/mmdet3d_plugin/models/temporal_memory/policies.py \
  projects/mmdet3d_plugin/models/detectors/repdetr3d.py
```

Expected: exit code 0.

- [ ] **Step 2: Run the complete focused suite again**

Run the Task 7 CPU verification command.

Expected: all tests PASS with zero failures.

- [ ] **Step 3: Verify the experiment row is no longer marked planned**

Run: `rg -n -A1 -B1 'counterfactual detection-degradation' RESULTS.md`

Expected: the row contains concrete commands and no `Planned:` marker.

- [ ] **Step 4: Commit any verification-only corrections**

If verification required corrections, commit only those corrections with:

```bash
git add tools/counterfactual_policy_targets.py tools/collect_depth_counterfactuals.py tools/train_temporal_policy_mlp.py projects/mmdet3d_plugin/models/temporal_memory/policy_targets.py projects/mmdet3d_plugin/models/temporal_memory/policies.py projects/mmdet3d_plugin/models/detectors/repdetr3d.py projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_4level_reset4_counterfactual_teacher_forcing_no_distill.py tests/test_counterfactual_policy_targets.py tests/test_collect_depth_counterfactuals.py tests/test_train_temporal_policy_mlp.py tests/test_temporal_memory_repdetr3d_routing.py tests/test_temporal_memory_configs.py RESULTS.md
git commit -m "fix: complete counterfactual policy verification"
```

If no correction was needed, do not create an empty commit.
