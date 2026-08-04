_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e_tm_depth_router_budget50_reset5.py']

# Train one temporal clip per GPU to avoid co-resident activation graphs from
# different routed depths (for example, L16 and L24) in the same minibatch.
work_dir = './work_dirs/repdetr3d_eva02_800_bs1_seq_24e_tm_depth_router_budget50_reset5'
data = dict(samples_per_gpu=1)

# Preserve the 24-epoch sample budget for the documented four-GPU launch.
num_iters_per_epoch = 28130 // (4 * 1)
evaluation = dict(interval=num_iters_per_epoch * 24)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)
runner = dict(type='IterBasedRunner', max_iters=num_iters_per_epoch * 24)
