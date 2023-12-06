Download [synthia (2).zip](https://sutdapac-my.sharepoint.com/:u:/g/personal/benjamin_luo_mymail_sutd_edu_sg/EXzLQluN-UVFstjYHwxlrQIBTyJFVTfIJi6jn_KYgC1wZw?e=Ljzimz) from onedrive as well as the stock cityscapes datasets and make sure both datasets are in data/

OR

If you have the stock SYNTHIA_RAND_CITYSCAPES dataset in data/, run the following scripts for conversion:

Splitting dataset into train-test-val splits ```python tools/dataset_converters/synthia_arrange.py data/synthia/```

Converting Synthia label images to the correct png format ```python tools/dataset_converters/synthia_label.py data/synthia/GT```

Converting Synthia labels to cityscape ones ```python tools/dataset_converters/synthia_synthia_2_cityscape.py data/synthia/GT```



### Training PMTrans

```python tools/train.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256.py```

Replace b0 with desired model size (b1, b2, b3, b4, b5)

### Testing PMTrans

```python tools/test.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-512x512.py work_dirs/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256/iter_[iteration].pth --show```

Replace b0 with desired model size (b1, b2, b3, b4, b5)