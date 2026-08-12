# Task 2 report

## Status

Completed Task 2 in the shared main checkout without touching unrelated dirty files. Added the high-resolution StreamPETR profiler script, the failing-then-passing unit test for summary formatting, and the planned 640x1600 teacher row plus reproducible commands in `RESULTS.md`.

## Commit

- `be85c314d87ed22159af058d6a95ebe043d2f2b3` — `feat: add StreamPETR high-resolution profiling`

## Verification

- Red step:
  - `TMPDIR=/tmp pytest -q tests/test_profile_gflops.py -k highres_profile_summary`
  - Result before implementation: `ModuleNotFoundError: No module named 'tools.profile_streampetr_highres'`
- Green step:
  - `TMPDIR=/tmp pytest -q tests/test_profile_gflops.py -k highres_profile_summary`
  - Result: `1 passed, 12 deselected in 1.16s`
- CLI validation:
  - `python tools/profile_streampetr_highres.py --help`
  - Result: exit code `0`; help includes `--config`, `--checkpoint`, `--output-dir`, and `--num-samples`

## Changed files

- `tools/profile_streampetr_highres.py`
- `tests/test_profile_gflops.py`
- `RESULTS.md`

## Notes

- The profiler rejects CPU execution with `RuntimeError("CUDA is required for high-resolution profiling")`.
- The summary written to `summary.json` reports `resolution`, peak allocated/reserved CUDA memory, and average backbone/head/total latency in milliseconds across the requested bounded sample count.
- The planned teacher row remains metric-blank as requested and includes the exact four-GPU train/eval commands.

## Concerns

- `python tools/profile_streampetr_highres.py --help` emits the repository's existing MMCV deprecation warning after the help text. The command still exits successfully and exposes the required arguments.

## Fix round 1

### Review items addressed

- Rewrote the planned `StreamPETR EVA02 640x1600 full-path teacher` `RESULTS.md` row so the command cell contains only the exact required 4-GPU train/eval commands.
- Added an injectable `ensure_cuda_available(...)` helper plus a unit test for CPU rejection without needing a GPU.
- Added `validate_consistent_resolution(...)` and enforced same-resolution validation across all measured samples, plus a unit test for mismatch rejection.

### Fix-round verification

- Red step:
  - `TMPDIR=/tmp pytest -q tests/test_profile_gflops.py -k 'highres_profile_requires_cuda or highres_profile_rejects_mixed_resolutions'`
  - Result before implementation: two `ImportError`s for missing `ensure_cuda_available` and `validate_consistent_resolution`
- Green step:
  - `TMPDIR=/tmp pytest -q tests/test_profile_gflops.py -k 'highres_profile_summary or highres_profile_requires_cuda or highres_profile_rejects_mixed_resolutions'`
  - Result: `3 passed, 12 deselected in 1.19s`
- CLI validation:
  - `python tools/profile_streampetr_highres.py --help`
  - Result: exit code `0`; required flags still present
