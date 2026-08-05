#!/usr/bin/env bash
set -euo pipefail

python tools/collect_l24_drift_reuse_risk.py \
  projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_reuse_k3.py \
  ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth \
  --output-dir outputs/l24_drift_reuse_risk \
  --num-scenes 100 \
  --targets-per-scene 20 \
  --seed 20260805 \
  --device cuda:0 \
  --cfg-options model.temporal_memory.keyframe_interval=2
