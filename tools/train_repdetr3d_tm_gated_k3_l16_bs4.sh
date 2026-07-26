#!/usr/bin/env bash
set -euo pipefail

bash tools/dist_train.sh \
  projects/configs/RepDETR3D/repdetr3d_eva02_800_bs4_seq_24e_tm_gated_k3_l16_no_distill.py \
  2 \
  --work-dir work_dirs/repdetr3d_eva02_800_bs4_seq_24e_tm_gated_k3_l16_no_distill
