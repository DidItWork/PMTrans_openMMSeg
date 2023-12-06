To prepare datasets for training, copy leftImg8bit/ and gtFine/ from cityscapes/ to your gta/ and synthia/ datasets


### Training PMTrans

```python tools/train.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256.py```

Replace b0 with desired model size (b1, b2, b3, b4, b5)

### Testing PMTrans

```python tools/test.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256.py work_dirs/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256/iter_[iteration].pth --show```

Replace b0 with desired model size (b1, b2, b3, b4, b5)
