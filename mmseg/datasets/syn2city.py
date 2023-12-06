# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import os.path as osp
from numpy.random import choice


@DATASETS.register_module()
class Syn2CityDataset(BaseSegDataset):
    """Synthia to CityScapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    # METAINFO = dict(
    #     classes=('void', 'sky', 'building', 'road', 'sidewalk', 'fence',
    #              'vegetation','pole','car','traffic sign', 'person',
    #              'bicycle', 'motorcycle', 'parking-slot', 'road-work',
    #              'traffic light', 'terrain', 'rider', 'truck', 'bus',
    #              'train', 'wall', 'lanemarking'),
    #     palette=[[0, 0, 0], [70, 130, 180], [70, 70, 70], [128, 64, 128],
    #              [244, 35, 232], [64, 64, 128], [107, 142, 35],
    #              [153, 153, 153], [0, 0, 142], [220, 220, 0], [220, 20, 60],
    #              [119, 11, 32], [0, 0, 230], [250, 170, 160], [128, 64, 64],
    #              [250, 170, 30], [152, 251, 152], [255, 0, 0], [0, 0, 70],
    #              [0, 60, 100], [0, 80, 100], [102, 102, 156], [102, 102, 156]])
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [64, 64, 128], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    def __init__(self,
                 img_suffix='.png',
                 target_suffix='_leftImg8bit.png'
                 seg_map_suffix='_labelTrainIds.png',
                 target_seg_map_suffix='_gtFine_labelTrainIds.png',
                 target_root = '',
                 target_prefix: dict = dict(img_path='', seg_map_path=''),
                 target_ann_file='',
                 **kwargs) -> None:
        self.target_prefix = target_prefix
        self.target_ann_file = target_ann_file
        self.target_root = target_root
        self.target_seg_map_suffix = target_seg_map_suffix
        self.target_suffix = target_suffix
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        
        
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        target_dir = osp.join(self.data_root, self.target_prefix.get('img_path', None))
        target_ann_dir = osp.join(self.target_root, self.target_prefix.get('seg_map_path', None))
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(
                    img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        
        #Target annotations
        if not osp.isdir(self.target_ann_file) and self.target_ann_file:
            assert osp.isfile(self.target_ann_file), \
                f'Failed to load `target_ann_file` {self.target_ann_file}'
            lines = mmengine.list_from_file(
                self.target_ann_file, backend_args=self.backend_args)
            for datainfo,line in zip(lines,data_list):
                img_name = line.strip()
                datainfo['target_path']=osp.join(img_dir, img_name + self.target_suffix)
                if target_ann_dir is not None:
                    seg_map = img_name + self.target_seg_map_suffix
                    datainfo['target_seg_map_path'] = osp.join(target_ann_dir, seg_map)
                datainfo['target_label_map'] = self.label_map
                datainfo['target_reduce_zero_label'] = self.reduce_zero_label
                datainfo['target_seg_fields'] = []
        else:
            _suffix_len = len(self.target_suffix)
            target_images  = list(fileio.list_dir_or_file(
                        dir_path=target_dir,
                        list_dir=False,
                        suffix=self.target_suffix,
                        recursive=True,
                        backend_args=self.backend_args))
            for datainfo in data_list:
                img = choice(target_images)
                datainfo['target_path']=osp.join(target_dir,img)
                if target_ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.target_seg_map_suffix
                    datainfo['target_seg_map_path'] = osp.join(target_ann_dir, seg_map)
                datainfo['target_label_map'] = self.label_map
                datainfo['target_reduce_zero_label'] = self.reduce_zero_label
                datainfo['target_seg_fields'] = []
        
        data_list = sorted(data_list, key=lambda x: x['img_path'])

        return data_list
