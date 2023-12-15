Make sure the data/ folder is organized as below:

```
data
├── gta
│   ├── images
│   ├── labels
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   ├── val
│   ├── leftImg8bit
│   │   ├── test
│   │   ├── train
│   │   ├── val
├── synthia
│   ├── Depth
│   ├── GT
│   │   ├── COLOR
│   │   ├── LABELS
│   ├── RGB
```

Splitting dataset into train-test-val splits ```python tools/dataset_converters/synthia_arrange.py data/synthia/```

Formating Synthia Images to be consistent with Cityscapes ones ```python tools/dataset_converters/synthia_label.py data/synthia/GT```


### Training PMTrans

```python tools/train.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256.py```

Replace b0 with desired model size (b1, b2, b3, b4, b5)

### Testing PMTrans

```python tools/test.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256.py work_dirs/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256/iter_[iteration].pth --show```

Replace b0 with desired model size (b1, b2, b3, b4, b5)
