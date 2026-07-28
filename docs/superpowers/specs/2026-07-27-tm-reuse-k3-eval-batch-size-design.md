# TM Reuse K3 Evaluation Batch Size Design

## Goal

Set the default per-GPU batch size to 4 for
`projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_reuse_k3.py`
without changing any baseline or other temporal-memory experiment config.

## Design

Add a config-local override:

```python
data = dict(samples_per_gpu=4)
```

MMCV config inheritance will merge this with the inherited `data` dictionary.
With four distributed workers, evaluation will use an effective batch size of
16 while retaining the inherited dataset definitions, pipelines, and worker
settings.

## Validation

Load the target config through MMCV and assert that the resolved
`data.samples_per_gpu` value is 4. Also confirm that the baseline config still
resolves to 2, proving the override is isolated to the requested experiment.

No unit test will be added because this is a declarative configuration-only
change; the user approved resolved-config validation instead.
