# Drift Quantile versus Depth Degradation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reproducible notebook relating adjacent-frame L24 representation drift to paired task degradation from forced L8, L12, and L16 counterfactual predictions.

**Architecture:** Use one self-contained notebook with pure functions for strict input validation, class-aware box matching, scene-bootstrap statistics, and plotting. Production mode consumes four official nuScenes result JSONs plus a shared-state provenance manifest; synthetic mode executes the complete analysis without a GPU or nuScenes data. Detector inference remains outside the notebook.

**Tech Stack:** Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, nuScenes devkit, Jupyter notebook JSON.

---

## File Structure

- Create: `notebooks/drift_quantile_vs_depth_degradation.ipynb` — the entire analysis and its synthetic smoke path.
- Read: `outputs/policy_signal_stats/val/policy_feature_similarity.csv` — existing L24 drift table.
- Generate: `outputs/drift_depth_analysis/` — previews, CSV summaries, and resolved configuration.
- Generate in production only: `paper/figures/fig_drift_vs_depth_degradation.pdf` and `paper/figures/fig_drift_vs_depth_degradation_scatter.pdf`.

Do not add a helper module, test module, or dependency. Compile all notebook cells and execute the synthetic path as the runnable check.

### Task 1: Notebook Contract and Input Loaders

**Files:**
- Create: `notebooks/drift_quantile_vs_depth_degradation.ipynb`

- [ ] **Step 1: Run the failing structural check**

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('notebooks/drift_quantile_vs_depth_degradation.ipynb')
assert path.is_file()
notebook = json.loads(path.read_text())
assert notebook['nbformat'] == 4
assert any('DRIFT_DEPTH_SYNTHETIC' in ''.join(c.get('source', []))
           for c in notebook['cells'])
PY
```

Expected: FAIL because the notebook does not exist.

- [ ] **Step 2: Create the notebook introduction and configuration**

The first markdown cell must state that the endpoint is a per-frame surrogate, not a decomposition of mAP or NDS, and that production predictions must share the same pre-frame memory state.

The first code cell must define:

```python
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path('..').resolve() if Path.cwd().name == 'notebooks' else Path.cwd()
SYNTHETIC = os.getenv('DRIFT_DEPTH_SYNTHETIC', '0') == '1'
OUTPUT_DIR = Path(os.getenv(
    'DRIFT_DEPTH_OUTPUT_DIR', ROOT / 'outputs/drift_depth_analysis'))
PAPER_DIR = Path(os.getenv(
    'DRIFT_DEPTH_PAPER_DIR', ROOT / 'paper/figures'))
CFG = {
    'manifest': ROOT / 'outputs/drift_depth_analysis/counterfactual_manifest.json',
    'drift_csv': ROOT / 'outputs/policy_signal_stats/val/policy_feature_similarity.csv',
    'data_root': ROOT / 'data/nuscenes',
    'version': 'v1.0-trainval',
    'loss_csv': ROOT / 'outputs/drift_depth_analysis/per_frame_detection_loss.csv',
    'depths': ('L8', 'L12', 'L16', 'L24'),
    'score_floor': 0.10,
    'center_gate_m': 4.0,
    'velocity_scale_mps': 10.0,
    'weights': {'confidence': 0.5, 'center': 0.2, 'size': 0.1,
                'yaw': 0.1, 'velocity': 0.1},
    'false_positive_penalty': 1.0,
    'false_negative_penalty': 1.0,
    'quantiles': 5,
    'bootstrap_repetitions': 2000,
    'seed': 20260804,
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not SYNTHETIC:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Add strict production loaders**

Implement these interfaces in one cell:

```python
def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())

def validate_manifest(manifest):
    required = {'shared_pre_frame_state', 'run_id',
                'canonical_state_advance_action', 'prediction_paths'}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f'manifest missing keys: {sorted(missing)}')
    if manifest['shared_pre_frame_state'] is not True:
        raise ValueError('counterfactual predictions must share pre-frame state')
    if set(manifest['prediction_paths']) != set(CFG['depths']):
        raise ValueError(f'prediction_paths must contain {CFG["depths"]}')
    return {
        depth: (Path(path) if Path(path).is_absolute() else ROOT / path)
        for depth, path in manifest['prediction_paths'].items()
    }

def standardize_prediction(record):
    return {
        'name': record['detection_name'],
        'translation': np.asarray(record['translation'], dtype=float),
        'size': np.asarray(record['size'], dtype=float),
        'rotation': np.asarray(record['rotation'], dtype=float),
        'velocity': np.asarray(record.get('velocity', [0.0, 0.0])[:2], dtype=float),
        'score': float(record['detection_score']),
    }

def load_predictions(path):
    payload = load_json(path)
    if not isinstance(payload.get('results'), dict):
        raise ValueError(f'{path} is not a nuScenes results JSON')
    return {
        token: [standardize_prediction(record) for record in records
                if float(record['detection_score']) >= CFG['score_floor']]
        for token, records in payload['results'].items()
    }

def validate_token_coverage(predictions):
    coverage = {depth: set(records) for depth, records in predictions.items()}
    reference = coverage['L24']
    mismatched = {depth: len(tokens ^ reference)
                  for depth, tokens in coverage.items() if tokens != reference}
    if mismatched:
        raise ValueError(f'depth token coverage differs from L24: {mismatched}')
    if not reference:
        raise ValueError('prediction artifacts contain no sample tokens')
    return reference

def load_drift(path, tokens):
    frame = pd.read_csv(path)
    required = {'token', 'scene_token', 'has_previous_frame',
                'similarity_L24_mean'}
    missing = required - set(frame)
    if missing:
        raise ValueError(f'drift table missing columns: {sorted(missing)}')
    frame = frame[frame['token'].isin(tokens)].copy()
    if frame['token'].duplicated().any():
        raise ValueError('drift table contains duplicate sample tokens')
    frame = frame[(frame['has_previous_frame'] == 1) &
                  frame['similarity_L24_mean'].notna()].copy()
    frame['l24_drift'] = 1.0 - frame['similarity_L24_mean'].astype(float)
    if not np.isfinite(frame['l24_drift']).all():
        raise ValueError('L24 drift contains non-finite values')
    return frame[['token', 'scene_token', 'l24_drift']]
```

Load GT annotation fields in the same global coordinates as nuScenes result JSONs:

```python
def load_ground_truth(data_root, version, tokens):
    from nuscenes import NuScenes
    from nuscenes.eval.detection.utils import category_to_detection_name
    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    result = {}
    for token in tokens:
        boxes = []
        for annotation_token in nusc.get('sample', token)['anns']:
            annotation = nusc.get('sample_annotation', annotation_token)
            name = category_to_detection_name(annotation['category_name'])
            if name is None:
                continue
            boxes.append({
                'name': name,
                'translation': np.asarray(annotation['translation'], dtype=float),
                'size': np.asarray(annotation['size'], dtype=float),
                'rotation': np.asarray(annotation['rotation'], dtype=float),
                'velocity': np.nan_to_num(
                    nusc.box_velocity(annotation_token)[:2]),
                'score': 1.0,
            })
        result[token] = boxes
    return result
```

- [ ] **Step 4: Re-run the structural check**

Expected: PASS with no output.

- [ ] **Step 5: Commit**

```bash
git add notebooks/drift_quantile_vs_depth_degradation.ipynb
git commit -m "feat: define drift depth notebook inputs"
```

### Task 2: Paired Matched-Prediction Error

**Files:**
- Modify: `notebooks/drift_quantile_vs_depth_degradation.ipynb`

- [ ] **Step 1: Add a deterministic synthetic dataset**

Create three scenes with ten frames each and drift from 0.01 to 0.30. Give L8, L12, L16, and L24 center-offset multipliers of 3.0, 2.0, 1.0, and 0.15 respectively, plus seeded noise with standard deviation 0.005. Use one `car` GT and one prediction per frame. Return `drift_frame`, `ground_truth`, and the four prediction mappings. In the inline assertions also call `frame_error([], [target], True)` and `frame_error([prediction], [], True)` and require both to equal one, exercising false-negative and false-positive penalties without distorting the synthetic trend.

- [ ] **Step 2: Add complete matching functions**

```python
def quaternion_yaw(rotation):
    w, x, y, z = rotation
    return np.arctan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))

def wrapped_angle_error(a, b):
    return abs(np.arctan2(np.sin(a - b), np.cos(a - b))) / np.pi

def pair_components(prediction, target):
    center_m = np.linalg.norm(
        prediction['translation'][:2] - target['translation'][:2])
    return {
        'confidence': 1.0 - prediction['score'],
        'center': center_m / CFG['center_gate_m'],
        'size': np.mean(np.abs(np.log(
            np.clip(prediction['size'], 1e-6, None) /
            np.clip(target['size'], 1e-6, None)))),
        'yaw': wrapped_angle_error(
            quaternion_yaw(prediction['rotation']),
            quaternion_yaw(target['rotation'])),
        'velocity': np.linalg.norm(
            prediction['velocity'] - target['velocity']) /
            CFG['velocity_scale_mps'],
    }

def frame_error(predictions, targets, include_confidence=True):
    denominator = max(len(predictions), len(targets), 1)
    total = 0.0
    matched_predictions, matched_targets = set(), set()
    names = sorted({box['name'] for box in predictions + targets})
    for name in names:
        pred_indices = [i for i, box in enumerate(predictions)
                        if box['name'] == name]
        target_indices = [i for i, box in enumerate(targets)
                          if box['name'] == name]
        if not pred_indices or not target_indices:
            continue
        matrix = np.full((len(pred_indices), len(target_indices)), 1e6)
        for row, pred_index in enumerate(pred_indices):
            for column, target_index in enumerate(target_indices):
                values = pair_components(
                    predictions[pred_index], targets[target_index])
                if values['center'] <= 1.0:
                    weights = CFG['weights']
                    keys = weights if include_confidence else [
                        key for key in weights if key != 'confidence']
                    matrix[row, column] = sum(
                        weights[key] * values[key] for key in keys)
        rows, columns = linear_sum_assignment(matrix)
        for row, column in zip(rows, columns):
            if matrix[row, column] >= 1e6:
                continue
            total += matrix[row, column]
            matched_predictions.add(pred_indices[row])
            matched_targets.add(target_indices[column])
    total += CFG['false_positive_penalty'] * (
        len(predictions) - len(matched_predictions))
    total += CFG['false_negative_penalty'] * (
        len(targets) - len(matched_targets))
    return total / denominator
```

Add `per_frame_errors(...)` to calculate `error_L8/L12/L16/L24`, box-only counterparts, and paired deltas against L24.

- [ ] **Step 3: Add inline endpoint assertions**

```python
if SYNTHETIC:
    drift_frame, ground_truth, predictions = synthetic_inputs()
    validate_token_coverage(predictions)
    per_frame = per_frame_errors(drift_frame, ground_truth, predictions)
    assert frame_error([], [], True) == 0.0
    assert (per_frame['degradation_L8'].mean() >
            per_frame['degradation_L12'].mean() >
            per_frame['degradation_L16'].mean())
```

- [ ] **Step 4: Compile all code cells**

```bash
python - <<'PY'
import json
from pathlib import Path
notebook = json.loads(Path(
    'notebooks/drift_quantile_vs_depth_degradation.ipynb').read_text())
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell-{index}', 'exec')
print('all code cells compile')
PY
```

Expected: `all code cells compile`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/drift_quantile_vs_depth_degradation.ipynb
git commit -m "feat: add paired depth degradation metric"
```

### Task 3: Quantiles and Scene-Grouped Uncertainty

**Files:**
- Modify: `notebooks/drift_quantile_vs_depth_degradation.ipynb`

- [ ] **Step 1: Assign five equal-frequency drift bins**

```python
def assign_quantiles(frame):
    frame = frame.copy()
    if frame['scene_token'].nunique() < 2:
        raise ValueError('analysis requires at least two scenes')
    if frame['l24_drift'].nunique() < CFG['quantiles']:
        raise ValueError('too few distinct drift values for requested quantiles')
    frame['drift_quantile'] = pd.qcut(
        frame['l24_drift'], CFG['quantiles'], labels=False,
        duplicates='raise') + 1
    return frame
```

- [ ] **Step 2: Add scene-block bootstrap**

For each of 2,000 seeded repetitions, resample scene tokens with replacement, concatenate all frames from each selected scene, and calculate each quantile mean and each depth's Spearman correlation. Do not resample individual frames.

- [ ] **Step 3: Add `summarize_endpoint(frame, prefix)`**

Return:

- quantile columns: `endpoint, depth, drift_quantile, drift_mean, degradation_mean, ci_low, ci_high, frames`;
- correlation columns: `endpoint, depth, spearman, ci_low, ci_high`.

Use 2.5 and 97.5 percentiles from scene-bootstrap samples. Run it for `degradation` and `box_degradation`.

- [ ] **Step 4: Compile all cells and commit**

```bash
python - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path('notebooks/drift_quantile_vs_depth_degradation.ipynb').read_text())
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell-{i}', 'exec')
print('all code cells compile')
PY
git add notebooks/drift_quantile_vs_depth_degradation.ipynb
git commit -m "feat: summarize degradation by drift quantile"
```

Expected: compilation passes before the commit.

### Task 4: Quantile, Scatter, and Sensitivity Figures

**Files:**
- Modify: `notebooks/drift_quantile_vs_depth_degradation.ipynb`

- [ ] **Step 1: Add the primary figure**

Implement `plot_quantiles(summary, correlations, output_stem)`. Plot L8/L12/L16 means with asymmetric 95% CI error bars, one consistent color per depth, a zero reference line, integer quantile ticks, and `rho` in each legend label. Save both PDF and 220-DPI PNG.

Call it once for the combined endpoint and once for the box-only endpoint under `OUTPUT_DIR`. The combined endpoint alone is copied to the paper figure directory.

- [ ] **Step 2: Add the scatter alternative**

Implement `plot_scatter(frame, summary, output_stem)` with one shared-axis subplot per shallow depth. Use transparent rasterized points, overlay quantile means, and fit `IsotonicRegression(out_of_bounds='clip')` to the sorted continuous drift/degradation pairs. Save PDF and PNG.

- [ ] **Step 3: Add detector-loss sensitivity**

If `CFG['loss_csv']` exists, require unique tokens and columns `loss_L8`, `loss_L12`, `loss_L16`, and `loss_L24`; merge one-to-one and summarize paired `loss_degradation_*` columns. If absent, print exactly:

```text
Optional detector-loss table not found; skipping detector-loss sensitivity.
```

Never fail merely because this optional table is absent.

- [ ] **Step 4: Compile all cells and commit**

```bash
python -m json.tool notebooks/drift_quantile_vs_depth_degradation.ipynb >/dev/null
git diff --check -- notebooks/drift_quantile_vs_depth_degradation.ipynb
git add notebooks/drift_quantile_vs_depth_degradation.ipynb
git commit -m "feat: plot drift versus depth degradation"
```

Expected: notebook JSON and whitespace checks pass.

### Task 5: Exports and End-to-End Synthetic Verification

**Files:**
- Modify: `notebooks/drift_quantile_vs_depth_degradation.ipynb`

- [ ] **Step 1: Add production/synthetic dispatch**

Synthetic mode calls `synthetic_inputs()`. Production mode loads and validates the manifest, four predictions, token-identical drift table, and GT. Both paths call the same matching, quantile, bootstrap, and plotting functions. Production must stop on missing artifacts and must never silently fall back to synthetic data.

- [ ] **Step 2: Export reproducibility artifacts**

Write:

- `per_frame_degradation.csv`;
- `quantile_summary.csv`;
- `correlation_summary.csv`;
- `analysis_config.json` with Paths converted to strings;
- `validation_summary.json` with run ID, canonical state action, shared-state flag, frame count, and scene count;
- primary and scatter PDFs/PNGs under `OUTPUT_DIR`.

In production only, also write the two requested PDFs under `PAPER_DIR`.

- [ ] **Step 3: Add a final non-empty artifact assertion**

```python
expected = [
    OUTPUT_DIR / 'per_frame_degradation.csv',
    OUTPUT_DIR / 'quantile_summary.csv',
    OUTPUT_DIR / 'correlation_summary.csv',
    OUTPUT_DIR / 'analysis_config.json',
    OUTPUT_DIR / 'validation_summary.json',
    OUTPUT_DIR / 'fig_drift_vs_depth_degradation.pdf',
    OUTPUT_DIR / 'fig_drift_vs_depth_degradation.png',
    OUTPUT_DIR / 'fig_drift_vs_depth_degradation_scatter.pdf',
    OUTPUT_DIR / 'fig_drift_vs_depth_degradation_scatter.png',
]
missing = [str(path) for path in expected
           if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise RuntimeError(f'missing or empty outputs: {missing}')
print(f'Validated {len(per_frame)} frames from '
      f'{per_frame.scene_token.nunique()} scenes; wrote {len(expected)} artifacts.')
```

- [ ] **Step 4: Execute every code cell in synthetic mode**

```bash
tm_output_dir=$(mktemp -d)
DRIFT_DEPTH_SYNTHETIC=1 \
DRIFT_DEPTH_OUTPUT_DIR="$tm_output_dir/output" \
DRIFT_DEPTH_PAPER_DIR="$tm_output_dir/paper" \
MPLBACKEND=Agg \
python - <<'PY'
import json
from pathlib import Path
notebook = json.loads(Path(
    'notebooks/drift_quantile_vs_depth_degradation.ipynb').read_text())
namespace = {}
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'cell-{index}', 'exec'), namespace)
PY
```

Expected final line: `Validated 30 frames from 3 scenes; wrote 9 artifacts.`

- [ ] **Step 5: Verify statistical direction**

```bash
TM_OUTPUT_DIR="$tm_output_dir/output" python - <<'PY'
import os
from pathlib import Path
import pandas as pd
root = Path(os.environ['TM_OUTPUT_DIR'])
correlations = pd.read_csv(root / 'correlation_summary.csv')
primary = correlations[correlations['endpoint'] == 'degradation']
assert set(primary['depth']) == {'L8', 'L12', 'L16'}
assert (primary['spearman'] > 0.8).all(), primary
quantiles = pd.read_csv(root / 'quantile_summary.csv')
assert set(quantiles['drift_quantile']) == {1, 2, 3, 4, 5}
print(primary[['depth', 'spearman']].to_string(index=False))
PY
```

Expected: every shallow depth has positive Spearman correlation above 0.8.

- [ ] **Step 6: Commit the verified notebook**

```bash
git add notebooks/drift_quantile_vs_depth_degradation.ipynb
git commit -m "test: verify drift depth analysis notebook"
```

## Production Input Manifest

The production notebook requires:

```json
{
  "shared_pre_frame_state": true,
  "run_id": "repdetr3d-eva02-val-shared-state-v1",
  "canonical_state_advance_action": "L24",
  "prediction_paths": {
    "L8": "outputs/drift_depth_analysis/predictions_L8/results_nusc.json",
    "L12": "outputs/drift_depth_analysis/predictions_L12/results_nusc.json",
    "L16": "outputs/drift_depth_analysis/predictions_L16/results_nusc.json",
    "L24": "outputs/drift_depth_analysis/predictions_L24/results_nusc.json"
  }
}
```

Leave `DRIFT_DEPTH_SYNTHETIC` unset for the production run. Missing production inputs must produce a clear error.
