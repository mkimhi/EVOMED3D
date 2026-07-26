_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_eva02_800_bs4_seq_24e'

num_gpus = 2
batch_size = 4
num_epochs = 24
num_iters_per_epoch = 28130 // (num_gpus * batch_size)

data = dict(samples_per_gpu=batch_size)

evaluation = dict(interval=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
