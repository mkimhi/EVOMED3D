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
