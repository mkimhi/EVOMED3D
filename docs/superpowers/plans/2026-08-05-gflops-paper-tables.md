# GFLOPs Paper Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure end-to-end GFLOPs for the paper's RepDETR3D paths at six-view `320 x 800`, document provenance, and fill only defensible paper cells.

**Architecture:** Pure helper functions own conversion, validation, and aggregation. A CLI reuses the counterfactual collector's runtime helpers, profiles a real batch with `fvcore.nn.FlopCountAnalysis`, and writes JSONL plus JSON. Dynamic-policy values use locally measured paths and complete validation action counts.

**Tech Stack:** Python 3.8, PyTorch, fvcore 0.1.5, MMCV/MMDetection3D, pytest, LaTeX.

---

## File structure

- Create `tools/gflops_utils.py` for dependency-light validation and aggregation.
- Create `tools/profile_gflops.py` for runtime construction, tracing, and JSON output.
- Create `tests/test_gflops_utils.py` and `tests/test_profile_gflops.py`.
- Create `paper/gflops_provenance.md`.
- Modify only GFLOPs-related portions of `paper/main.tex`.

### Task 1: Pure calculations

**Files:**
- Create: `tests/test_gflops_utils.py`
- Create: `tools/gflops_utils.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from tools.gflops_utils import (macs_to_gflops, summarize_frames,
                                validate_input_shape,
                                validate_unsupported_ops,
                                weighted_path_gflops)

def test_two_flops_per_mac():
    assert macs_to_gflops(2_300_500_000_000) == pytest.approx(4601.0)

def test_expected_image_shape():
    assert validate_input_shape((1, 6, 3, 320, 800)) == (1, 6, 3, 320, 800)
    with pytest.raises(ValueError, match='expected image shape'):
        validate_input_shape((1, 6, 3, 800, 1333))

def test_weighted_paths():
    assert weighted_path_gflops({'L8': 100, 'L24': 400},
                                {'L8': 3, 'L24': 1}) == pytest.approx(175)

def test_frame_summary():
    with pytest.raises(ValueError, match='no frame measurements'):
        summarize_frames([])
    value = summarize_frames([{'action': 'L8', 'macs': 50e9},
                              {'action': 'L24', 'macs': 200e9}])
    assert value['mean_gflops'] == pytest.approx(250)
    assert value['action_counts'] == {'L8': 1, 'L24': 1}

def test_unsupported_compute_is_rejected():
    with pytest.raises(ValueError, match='unsupported compute operators'):
        validate_unsupported_ops({'aten::einsum': 2})
    assert validate_unsupported_ops({'aten::reshape': 3}) == {'aten::reshape': 3}
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_gflops_utils.py`

Expected: import failure because `tools.gflops_utils` does not exist.

- [ ] **Step 3: Implement the minimal functions**

```python
from collections import Counter

EXPECTED_IMAGE_SHAPE = (1, 6, 3, 320, 800)
IGNORED_UNSUPPORTED_OPS = {'aten::reshape', 'aten::view', 'aten::permute',
                           'aten::transpose', 'aten::contiguous',
                           'aten::slice', 'aten::select', 'aten::index',
                           'aten::unsqueeze', 'aten::squeeze', 'aten::detach'}

def macs_to_gflops(macs):
    if float(macs) < 0:
        raise ValueError('MAC count must be non-negative')
    return float(macs) * 2 / 1e9

def validate_input_shape(shape, expected=EXPECTED_IMAGE_SHAPE):
    shape = tuple(int(x) for x in shape)
    if shape != tuple(expected):
        raise ValueError(f'expected image shape {tuple(expected)}, got {shape}')
    return shape

def validate_unsupported_ops(unsupported, ignored=IGNORED_UNSUPPORTED_OPS):
    unsupported = {k: int(v) for k, v in unsupported.items() if int(v) > 0}
    material = sorted(set(unsupported) - set(ignored))
    if material:
        raise ValueError(f'unsupported compute operators: {material}')
    return unsupported

def weighted_path_gflops(path_gflops, action_counts):
    counts = {k: int(v) for k, v in action_counts.items() if int(v) > 0}
    if not counts:
        raise ValueError('action counts are empty')
    missing = sorted(set(counts) - set(path_gflops))
    if missing:
        raise ValueError(f'missing path GFLOPs for actions: {missing}')
    return sum(float(path_gflops[k]) * v for k, v in counts.items()) / sum(counts.values())

def summarize_frames(rows):
    rows = list(rows)
    if not rows:
        raise ValueError('no frame measurements')
    mean_macs = sum(float(row['macs']) for row in rows) / len(rows)
    return {'sample_count': len(rows), 'mean_macs': mean_macs,
            'mean_gflops': macs_to_gflops(mean_macs),
            'action_counts': dict(Counter(row['action'] for row in rows))}
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest -q tests/test_gflops_utils.py` (expected: 5 passed).

```bash
git add tools/gflops_utils.py tests/test_gflops_utils.py
git commit -m "feat: add GFLOPs aggregation utilities"
```

### Task 2: Synthetic fvcore contract

**Files:**
- Create: `tests/test_profile_gflops.py`
- Create: `tools/profile_gflops.py`

- [ ] **Step 1: Write the failing tests**

```python
import torch
from torch import nn
from tools.profile_gflops import analyze_module, parse_args

def test_linear_trace():
    result = analyze_module(nn.Linear(4, 3, bias=False),
                            (torch.zeros(2, 4),),
                            image_shape=(1, 6, 3, 320, 800))
    assert result['macs'] == 24
    assert result['gflops'] == 48 / 1e9
    assert result['unsupported_ops'] == {}

def test_cli_defaults(tmp_path):
    args = parse_args(['config.py', 'model.pth', '--output-dir', str(tmp_path)])
    assert (args.device, args.num_samples, args.forced_action) == ('cuda:0', 1, None)
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_profile_gflops.py`

Expected: import failure because `tools.profile_gflops` does not exist.

- [ ] **Step 3: Implement parser and analyzer**

Use `FlopCountAnalysis(module, inputs)`, call `analysis.total()`, collect `unsupported_ops()` and `by_operator()`, convert with `macs_to_gflops`, validate the image shape, and reject unreviewed unsupported operators. The parser accepts positional config/checkpoint and output directory, device, sample count, forced action, data split, and MMCV config overrides.

- [ ] **Step 4: Verify GREEN and commit**

Run: `pytest -q tests/test_profile_gflops.py` (expected: 2 passed).

```bash
git add tools/profile_gflops.py tests/test_profile_gflops.py
git commit -m "feat: add fvcore GFLOPs profiler contract"
```

### Task 3: Real RepDETR3D tracing

**Files:**
- Modify: `tools/profile_gflops.py`
- Modify: `tests/test_profile_gflops.py`

- [ ] **Step 1: Add failing provenance-output tests**

Test that `frame_record` preserves index, tokens, requested/observed action, forced flag, image shape, MACs, GFLOPs, unsupported ops, and operator totals. Test that `append_jsonl` writes exactly one JSON record.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_profile_gflops.py -k 'frame_record or append_jsonl'`

Expected: import failure for the missing functions.

- [ ] **Step 3: Implement runtime with existing helpers**

Reuse `_import_config_plugin`, `forced_action`, `select_dataset_cfg`, and `validate_model` from `tools/collect_depth_counterfactuals.py`. Build a batch-1 non-shuffled data loader and configured model, load the checkpoint, and call `eval()`.

Use MMCV `scatter_kwargs` to obtain device-side arguments. A `DetectorTraceWrapper(nn.Module)` stores the model and non-image arguments; its only traced argument is the image tensor and its forward calls the detector with `return_loss=False` and `rescale=True`. Validate the real tensor shape before tracing. Record observed controller action, reject unexpected action mismatch, append completed frames to JSONL, and write a summary with paths, hashes, scope, convention, frame mean, action counts, and operator totals.

```python
class DetectorTraceWrapper(nn.Module):
    def __init__(self, model, model_kwargs):
        super().__init__()
        self.model = model
        self.model_kwargs = model_kwargs

    def forward(self, img):
        return self.model(return_loss=False, rescale=True,
                          img=img, **self.model_kwargs)

def frame_record(index, token, scene_token, requested_action,
                 observed_action, forced, analysis):
    row = dict(analysis)
    row.update(index=int(index), sample_token=str(token),
               scene_token=str(scene_token),
               requested_action=requested_action,
               action=str(observed_action), forced=bool(forced))
    return row

def append_jsonl(path, record):
    with Path(path).open('a') as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write(chr(10))
```

- [ ] **Step 4: Verify and commit**

```bash
pytest -q tests/test_gflops_utils.py tests/test_profile_gflops.py
python -m py_compile tools/gflops_utils.py tools/profile_gflops.py
git add tools/profile_gflops.py tests/test_profile_gflops.py
git commit -m "feat: profile RepDETR3D GFLOPs"
```

### Task 4: Smoke test and operator coverage

**Files:**
- Modify only if coverage requires a reviewed rule: profiler files and tests.
- Produce: `outputs/gflops/smoke*/frames.jsonl`
- Produce: `outputs/gflops/smoke*/summary.json`

- [ ] **Step 1: Check GPU and checkpoint**

```bash
nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader
test -f work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_feature_noise_no_distill/iter_42192.pth
```

- [ ] **Step 2: Run one L24 profile**

```bash
CUDA_VISIBLE_DEVICES=0 python tools/profile_gflops.py projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_online_offline_teacher_forcing_no_distill.py work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_feature_noise_no_distill/iter_42192.pth --output-dir outputs/gflops/smoke --num-samples 1 --forced-action L24 --device cuda:0 --cfg-options model.temporal_memory.max_frames_between_full=4
```

Expected: positive MACs, observed L24, shape `[1,6,3,320,800]`, and no unreviewed unsupported compute operator.

- [ ] **Step 3: Resolve observed unsupported operators test-first**

For each material operator, add a synthetic failing test with a hand-computed MAC count before adding an fvcore op handle. Only zero-MAC reshape/index/metadata operations may enter the ignore set. Repeat the smoke run until coverage passes.

- [ ] **Step 4: Repeat into `outputs/gflops/smoke_repeat`**

Compare `mean_macs` in both summary files and require exact equality.

- [ ] **Step 5: Commit coverage rules if needed**

```bash
git add tools/gflops_utils.py tools/profile_gflops.py tests/test_gflops_utils.py tests/test_profile_gflops.py
git diff --cached --quiet || git commit -m "fix: cover RepDETR3D profiling operators"
```

### Task 5: Measure paths and write provenance

**Files:**
- Produce: `outputs/gflops/l24/summary.json`
- Produce: `outputs/gflops/l8/summary.json`
- Produce: `outputs/gflops/l12/summary.json`
- Produce: `outputs/gflops/l16/summary.json`
- Produce: `outputs/gflops/reuse/summary.json`
- Create: `paper/gflops_provenance.md`

- [ ] **Step 1: Profile L24/L8/L12/L16**

Repeat the Task 4 command with `--num-samples 2`, a separate output directory, and the corresponding forced action. Two frames initialize memory before a partial action.

- [ ] **Step 2: Profile reuse**

```bash
CUDA_VISIBLE_DEVICES=0 python tools/profile_gflops.py projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_reuse_k3.py ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth --output-dir outputs/gflops/reuse --num-samples 2 --device cuda:0
```

- [ ] **Step 3: Extract complete validation action counts**

Use the final controller report from the exact evaluation log for each paper policy. If a row lacks complete counts, run its existing test command with `model.tm_debug_latency=True`. Never infer action counts from latency.

- [ ] **Step 4: Compute dynamic averages**

Call `weighted_path_gflops` using locally measured paths and recorded action counts, including forced L24 refreshes. Label these results `local-weighted`.

- [ ] **Step 5: Create and validate provenance**

For every row record table, GFLOPs, evidence tier, resolution, cameras, convention, config, checkpoint or citation, aggregation, unsupported ops, and notes. Record published ToC3D StreamPETR as 4,601 and ToC3D-Faster as 3,131 GFLOPs. Record ToC3D-Fast and SToRe3D-1/3 and 1/10 as `n/r`; never estimate them from keep ratios.

Run a short Python assertion that every numeric local row references an existing summary JSON and each action-count total matches the cited evaluation sample count.

- [ ] **Step 6: Commit provenance**

```bash
git add paper/gflops_provenance.md
git commit -m "docs: record GFLOPs measurement provenance"
```

### Task 6: Update and build the paper

**Files:**
- Modify: `paper/main.tex`

- [ ] **Step 1: Preserve the user's current diff**

Run: `git diff -- paper/main.tex`. Confine changes to GFLOPs cells, relevant column definitions and captions, and the experimental protocol.

- [ ] **Step 2: Fill verified values**

Fill the main and depth/policy tables, and add GFLOPs only to inference-efficiency updater/policy tables. Keep unavailable related work as `n/r`. Add this protocol text:

```latex
We report end-to-end GFLOPs per six-camera timestamp at
$320\times800$ and batch size 1, counting one multiply--add as two
floating-point operations. Dynamic-policy costs are validation-set
averages that include forced full-depth refreshes.
```

- [ ] **Step 3: Check structure and build**

```bash
rg -n "GFLOPs|MISSING|n/r" paper/main.tex
git diff --check -- paper/main.tex paper/gflops_provenance.md
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: no blank GFLOPs cell in an inference-efficiency table and LaTeX exits 0.

- [ ] **Step 4: Run final verification**

```bash
pytest -q tests/test_gflops_utils.py tests/test_profile_gflops.py
git diff --check
git status --short
```

- [ ] **Step 5: Commit without staging unrelated files**

```bash
git add paper/main.tex
git commit -m "docs: report detector GFLOPs"
```

Do not use `git add -A`; the working tree contains unrelated user files.
