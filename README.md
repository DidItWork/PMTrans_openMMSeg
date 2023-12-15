# PMFormer: Patch-Mix Transformer for Domain-Adaptive Semantic Segmentation

## Get Started

Please see [Overview](docs/en/overview.md) for the general introduction of MMSegmentation.

## Setup Datasets

Download [synthia (2).zip](https://sutdapac-my.sharepoint.com/:u:/g/personal/benjamin_luo_mymail_sutd_edu_sg/EXzLQluN-UVFstjYHwxlrQIBTyJFVTfIJi6jn_KYgC1wZw?e=Ljzimz) from onedrive as well as the stock cityscapes datasets and make sure both datasets are in data/

OR

If you have the stock SYNTHIA_RAND_CITYSCAPES dataset in data/, run the following scripts for conversion:

Splitting Synthia dataset into train-test-val splits ```python tools/dataset_converters/synthia_arrange.py data/synthia/```

Formating Synthia Images to be consistent with Cityscapes ones ```python tools/dataset_converters/synthia_2_cityscape.py data/synthia/GT```

The final folder structure should look like this:
```none
DAFormer
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
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
python tools/train.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256.py
```
```
Note: You can also replace b0 with desired model size (b1, b2, b3, b4, b5)
```


## Testing PMTrans

```bash
python tools/test.py configs/segformer/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256.py work_dirs/pmtrans_mit-b0_8xb1-40k_synthia2cityscapes-256x256/iter_[iteration].pth --show
```
