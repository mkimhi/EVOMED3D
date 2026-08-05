# Axis safety fix report

## Root cause

`zero_free_data_limits` classified a minimum of exactly zero as positive-only
data. This replaced an observed zero with the smallest positive float, clipping
zero-valued outcomes; with an all-zero axis, applying the corresponding upper
branch made the range invalid.

## Change

Only strictly positive minima and strictly negative maxima now use the
zero-avoiding bounds. Observed zero values retain ordinary data-driven padding,
so they are visible in both mixed and all-zero ranges. Strictly positive data
continues to use a non-forced-zero informative lower limit.

## Regression coverage and verification

- Added mixed-zero and all-zero `zero_free_data_limits` regression tests.
- Focused helper tests: `3 passed`.
- Focused project suite: `pytest -q tests/test_l24_drift_reuse_risk.py tests/test_collect_depth_counterfactuals.py` — `26 passed`.
- Syntax validation: `python -m py_compile tools/l24_drift_reuse_risk.py tools/collect_l24_drift_reuse_risk.py` — passed.
- `git diff --check` — passed.
- Documented synthetic `jupyter nbconvert --execute` smoke run — passed; produced the executed notebook and all seven expected smoke artifacts.
