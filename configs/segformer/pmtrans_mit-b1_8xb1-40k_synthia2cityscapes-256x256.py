_base_ = ['./pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
