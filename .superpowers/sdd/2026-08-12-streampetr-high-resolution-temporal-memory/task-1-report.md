# Task 1 Report: High-resolution StreamPETR teacher config and static tests

## Status

DONE_WITH_CONCERNS

## Commits

- `feat: add high-resolution StreamPETR teacher config` (current Task 1 commit at handoff; see `git log --oneline -1`)

## Summary

Implemented the new camera-only 640x1600 StreamPETR EVA teacher config at `projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py` and added a static regression/config-merge test in `tests/test_temporal_memory_config.py`.

## TDD record

1. Added `test_streampetr_eva02_highres_teacher_config_is_camera_only_and_checkpoint_safe`.
2. Ran `TMPDIR=/tmp pytest -q tests/test_temporal_memory_config.py -k highres_teacher`.
3. Observed the initial failure because `projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py` did not exist.
4. Added the config as a narrow override of the EVA 800 config.
5. Extended the same test to verify the merged config pipelines actually carry `final_dim=(640, 1600)` after `Config.fromfile(...)`.
6. Re-ran the test and observed a second failure because inherited base pipelines still resolved to `(320, 800)`.
7. Fixed the config by overriding the merged train/test/eval pipelines and data/evaluation pipeline references so the loaded config truly uses the high-resolution augmentation settings.
8. Re-ran the test and it passed.

## Commands run

### Red

```bash
TMPDIR=/tmp pytest -q tests/test_temporal_memory_config.py -k highres_teacher
```

Initial failure:

```text
FileNotFoundError: [Errno 2] No such file or directory: '/home/moshekimhi_google_com/EVOMED3D/projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py'
```

Second failure after first config draft:

```text
assert (320, 800) == (640, 1600)
```

### Green

```bash
TMPDIR=/tmp pytest -q tests/test_temporal_memory_config.py -k highres_teacher
```

Output:

```text
.                                                                        [100%]
1 passed, 6 deselected, 3 warnings in 1.93s
```

### Python compilation

```bash
python -m py_compile projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py tests/test_temporal_memory_config.py
```

Output:

```text
[no output]
```

### Config parse

```bash
python - <<'PY'
from mmcv import Config
cfg = Config.fromfile('projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py')
print('train_final_dim', cfg.train_pipeline[4]['data_aug_conf']['final_dim'])
print('test_final_dim', cfg.test_pipeline[1]['data_aug_conf']['final_dim'])
print('eval_final_dim', cfg.evaluation['pipeline'][1]['data_aug_conf']['final_dim'])
PY
```

Output:

```text
train_final_dim (640, 1600)
test_final_dim (640, 1600)
eval_final_dim (640, 1600)
```

## Changed files

- `projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py`
- `tests/test_temporal_memory_config.py`
- `.superpowers/sdd/2026-08-12-streampetr-high-resolution-temporal-memory/task-1-report.md`

## Concerns

1. Final verification in this completion was limited to the requested static config test, Python compilation, and config parsing; it does not re-verify checkpoint-load/runtime behavior.

---

## Fix round 1

### Status

DONE_WITH_CONCERNS

### Review items addressed

1. Ran the exact brief smoke command and fixed its root causes so it now completes from the shared checkout without extra environment variables.
2. Extended the static high-resolution config test to assert:
   - `num_iters_per_epoch == 28130 // 4`
   - `runner.max_iters == num_epochs * num_iters_per_epoch`
   - `checkpoint_config.interval == num_iters_per_epoch`
3. Added a regression check that the high-resolution config does not inherit `WandbLoggerHook`, because that was the remaining non-interactive smoke blocker after fixing launcher imports.

### Root-cause findings

1. `python tools/train.py ...` failed with `ModuleNotFoundError: No module named 'projects'` because running the script from `tools/` did not put the repository root on `sys.path`, so plugin import resolution via `projects.mmdet3d_plugin` failed.
2. After fixing that launcher issue, the exact smoke command still failed in `WandbLoggerHook` with `No API key configured` because the new high-resolution config inherited the base EVA W&B logger hooks unchanged.

### Fixes

- Updated `tools/train.py` to insert the repository root into `sys.path` before plugin imports.
- Updated `projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py` to override `log_config` with only `TextLoggerHook` and `TensorboardLoggerHook`.
- Extended `tests/test_temporal_memory_config.py` with the requested iteration/checkpoint assertions and the non-W&B logger regression check.

### Commands run

#### Red: exact brief smoke command before fixes

```bash
python tools/train.py projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py --cfg-options runner.max_iters=1 data.workers_per_gpu=0 --work-dir work_dirs/tm_streampetr_1600_config_smoke
```

Output before launcher fix:

```text
ModuleNotFoundError: No module named 'projects'
```

Output after launcher fix but before config logging fix:

```text
wandb.errors.errors.UsageError: No API key configured. Use `wandb login` to log in.
```

#### Red: extended static test before config logging fix

```bash
TMPDIR=/tmp pytest -q tests/test_temporal_memory_config.py -k highres_teacher
```

Output:

```text
KeyError: 'log_config'
```

#### Green: extended static test

```bash
TMPDIR=/tmp pytest -q tests/test_temporal_memory_config.py -k highres_teacher
```

Output:

```text
.                                                                        [100%]
1 passed, 6 deselected, 3 warnings in 2.22s
```

#### Green: exact brief smoke command after fixes

```bash
python tools/train.py projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py --cfg-options runner.max_iters=1 data.workers_per_gpu=0 --work-dir work_dirs/tm_streampetr_1600_config_smoke
```

Key output excerpts:

```text
2026-08-12 20:50:37,722 - mmdet - INFO - Config:
...
final_dim=(640, 1600)
...
2026-08-12 20:50:59,412 - mmdet - INFO - load checkpoint from local path: ./ckpts/eva02_L_coco_det_sys_o365_remapped.pth
2026-08-12 20:51:00,590 - mmdet - WARNING - The model and loaded state dict do not match exactly
size mismatch for img_backbone.rope_glb.freqs_cos: copying a param with shape torch.Size([9216, 64]) from checkpoint, the shape in current model is torch.Size([400, 64]).
size mismatch for img_backbone.rope_glb.freqs_sin: copying a param with shape torch.Size([9216, 64]) from checkpoint, the shape in current model is torch.Size([400, 64]).
2026-08-12 20:51:00,592 - mmdet - INFO - workflow: [('train', 1)], max: 1 iters
2026-08-12 20:51:07,093 - mmdet - INFO - Saving checkpoint at 1 iterations
```

Exit code: `0`

### Files changed in fix round 1

- `tools/train.py`
- `projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py`
- `tests/test_temporal_memory_config.py`
- `.superpowers/sdd/2026-08-12-streampetr-high-resolution-temporal-memory/task-1-report.md`

### Concerns

1. The exact brief smoke command now completes, but EVA checkpoint loading still reports the expected/non-fatal rotary-position mismatch warnings at 640×1600 (`rope_glb.freqs_cos/sin` shape `9216 -> 400`), so startup is checkpoint-compatible rather than warning-free.
