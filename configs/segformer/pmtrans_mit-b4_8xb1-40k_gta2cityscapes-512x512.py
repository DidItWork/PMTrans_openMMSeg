_base_ = ['./pmtrans_mit-b0_8xb1-40k_gta2cityscapes-512x512.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
