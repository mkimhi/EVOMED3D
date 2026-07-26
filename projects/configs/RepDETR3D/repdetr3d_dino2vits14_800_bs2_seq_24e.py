_base_ = ['./repdetr3d_eva02_800_bs2_seq_24e.py']

work_dir = './work_dirs/repdetr3d_dino2vits14_800_bs2_seq_24e'

dino_sim_fpn = dict(
    scale_factors=[4, 2, 1, 0.5],
    in_channels=384,
    out_channels=256,
    out_indices=[2, 3, 4, 5],
)

model = dict(
    img_backbone=dict(
        type='EVAViT',
        img_size=320,
        patch_size=14,
        window_size=14,
        global_window_size=0,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        window_block_indexes=tuple(range(12)),
        sim_fpn=dino_sim_fpn,
        qkv_bias=True,
        drop_path_rate=0.1,
        with_cp=True,
        flash_attn=False,
        pretrain_img_size=518,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

load_from = './ckpts/dinov2_vits14_reg4_repdetr3d_remapped.pth'
