Download [https://sutdapac-my.sharepoint.com/:u:/g/personal/benjamin_luo_mymail_sutd_edu_sg/EXzLQluN-UVFstjYHwxlrQIBTyJFVTfIJi6jn_KYgC1wZw?e=Ljzimz](synthia (2).zip) from onedrive and make sure both datasets are in data/

### Training PMTrans

```python tools/train.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-512x512.py```

Replace b0 with desired model size (b1, b2, b3, b4, b5)

### Testing PMTrans

```python tools/test.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-512x512.py work_dirs/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-512x512/iter_[iteration].pth --show```
