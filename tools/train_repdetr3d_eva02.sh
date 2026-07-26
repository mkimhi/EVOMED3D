#!/usr/bin/env bash
set -euo pipefail

GPUS=${1:-2}
shift || true

export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

bash tools/dist_train.sh \
  projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e.py \
  "$GPUS" \
  --work-dir work_dirs/repdetr3d_eva02_800_bs2_seq_24e \
  "$@"
