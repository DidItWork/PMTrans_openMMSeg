# PMFormer: Patch-Mix Transformer for Domain-Adaptive Semantic Segmentation

## Get Started

Please see [Overview](docs/en/overview.md) for the general introduction of MMSegmentation.

## Setup Datasets

After unzipping SYNTHIA_RAND_CITYSCAPES dataset into data/synthia, run the following scripts for conversion:

Splitting Synthia Images into Train-Test-Val splits 
```bash
python tools/dataset_converters/synthia_arange.py data/synthia/
```

Formating Synthia Images to be consistent with Cityscapes ones 
```bash
python tools/dataset_converters/synthia_2_cityscape.py data/synthia/GT
```

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

## Pseudo Labelling

```bash
python target_labelling.py data/cityscapes/leftImg8bit
```

```
By default, the script looks in work_dirs/segformer_mit-b5_8xb1-40k_synthia-512x512/iter_40000.pth for the model file.
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

## File Locations

- The model and training losses can be found in [pmtrans](mmseg/models/segmentors/pmtrans.py)
- The dataset configurations can be found in [configs/\_base\_/datasets/](configs/_base_/datasets/)
- The training configurations are found in [configs/segformer/](configs/segformer/)
