# L24 drift versus naive-reuse K=2 risk analysis

This analysis uses the full final L24 backbone/neck representation and the deployed RepDETR3D EVA-02 naive-reuse K=2 detector. It selects exactly 20 non-keyframe targets in each of 100 validation scenes (2,000 targets), measures delta-t=1 and delta-t=2 cosine drift, and relates each to absolute GT prediction and box error.

## Collect the fixed study

Run from the repository root on a CUDA-capable machine with the checkpoint and nuScenes data available:

```bash
bash tools/run_l24_drift_reuse_risk.sh
```

The script resolves to this exact collector invocation:

```bash
python tools/collect_l24_drift_reuse_risk.py \
  projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_reuse_k3.py \
  ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth \
  --output-dir outputs/l24_drift_reuse_risk \
  --num-scenes 100 --targets-per-scene 20 --seed 20260805 --device cuda:0 \
  --cfg-options model.temporal_memory.keyframe_interval=2
```

The collector refuses an already non-empty output directory. Do not remove or replace an existing study output unless that action is deliberate and approved.

## Render the analysis

After collection, execute the notebook from the repository root:

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/l24_drift_vs_naive_reuse_gt_loss.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --output outputs/l24_drift_reuse_risk/executed_notebook.ipynb
```

The production notebook has no synthetic fallback. It fails clearly if `collection_manifest.json` or any referenced collector artifact is absent, has the wrong base-L24/K=2 provenance, or does not cover exactly 2,000 unique targets from 100 scenes. It validates all collector artifacts before loading nuScenes GT.

Expected output files under `outputs/l24_drift_reuse_risk/` are:

- `per_frame_gt_loss.csv`
- `quantile_summary.csv`
- `correlation_summary.csv`
- `analysis_config.json`
- `validation_summary.json`
- `fig_l24_drift_vs_naive_reuse_gt_loss.pdf`
- `fig_l24_drift_vs_naive_reuse_gt_loss.png`

The production notebook also copies the PDF and 300-dpi PNG figure to `paper/figures/` using the same basename. The figure is a 2-by-2 panel: prediction/box loss by delta-t=1/delta-t=2, with density-aware rasterized scatter, isotonic trend, quintile means with scene-bootstrap 95% CIs, and bootstrap Spearman intervals (2,000 complete-scene resamples).

## Synthetic notebook smoke check

This is only an execution check; it generates deterministic synthetic inputs, uses 20 bootstrap resamples for speed, and does not touch `paper/figures/`. Production analysis always uses 2,000 complete-scene resamples.

```bash
TM_L24_REUSE_RISK_SMOKE=1 \
TM_L24_REUSE_RISK_OUTPUT_DIR=outputs/l24_drift_reuse_risk_smoke \
jupyter nbconvert --to notebook --execute \
  notebooks/l24_drift_vs_naive_reuse_gt_loss.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --output outputs/l24_drift_reuse_risk_smoke/executed_notebook.ipynb
```

## Interpretation boundary

Absolute GT loss demonstrates K=2 deployment-risk association and can include intrinsic frame difficulty; it does not isolate the causal excess error due to reuse. Do not describe this figure as a K=2-minus-full counterfactual degradation result.
