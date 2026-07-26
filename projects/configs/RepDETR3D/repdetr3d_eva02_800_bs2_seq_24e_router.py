_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

# Fixed-token RepDETR3D routing experiment. Patch size stays unchanged; the
# SparseHead hook preserves dense feature geometry required by deformable
# attention and fills dropped token positions with ambient context.
work_dir = './work_dirs/repdetr3d_eva02_800_bs2_seq_24e_router'

model = dict(
    pts_bbox_head=dict(
        routing_mode='ambient_pool',
        routing_keep_ratio=0.5,
        routing_keep_ratio_stopped=0.5,
        routing_keep_ratio_moderate=0.6,
        routing_keep_ratio_fast=0.7,
        routing_stopped_speed=0.5,
        routing_high_speed=12.0,
        routing_log_stats=False,
    ),
)
