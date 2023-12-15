# PMFormer: Patch-Mix Transformer for Domain-Adaptive Semantic Segmentation

## Get Started

Please see [Overview](docs/en/overview.md) for the general introduction of MMSegmentation.

## Setup Datasets

After unzipping SYNTHIA_RAND_CITYSCAPES dataset into data/synthia, run the following scripts for conversion:

Formating Synthia Images to be consistent with Cityscapes ones ```python tools/dataset_converters/synthia_2_cityscape.py data/synthia/GT```

The final folder structure should look like this:
```none
DAFormer
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── test
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── test
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```


## Training PMTrans

```bash
python tools/train.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-512x512.py
```
```
Note: You can also replace b0 with desired model size (b1, b2, b3, b4, b5)
```


## Testing PMTrans

```bash
python tools/test.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-512x512.py work_dirs/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-512x512/iter_[iteration].pth --show
```
