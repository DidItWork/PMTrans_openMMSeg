import argparse
import os.path as osp
import numpy as np
from imageio.v2 import imread, imwrite
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)
# from mmcv import imread, imwrite

from mmengine.utils import scandir

mappings = [255, #0
                10, #1
                2, #2
                0, #3
                1, #4
                4, #5
                8, #6
                5, #7
                13, #8
                7, #9
                11, #10
                18, #11
                17, #12
                0, #13
                255, #14
                6, #15
                9, #16
                12, #17
                14, #18
                15, #19
                16, #20
                3, #21
                0 #22
                ]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Formating raw SYNTHIA_RAND_CITYSCAPES labels')
    parser.add_argument('synthia_GT_path', help='synthia ground truth data path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args

def convert_labels(img_path):
    synthia_img = np.asarray(imread(img_path, format="PNG-FI"),np.uint8)[:,:,0]
    cityscape_img = mappings[synthia_img] 
    imwrite(img_path.replace('LABELS','_cityscapesLabels'),cityscape_img)

def main():

    args = parse_args()

    gt_dir = args.synthia_GT_path
    
    img_paths = []
    for img_path in scandir(gt_dir, 'LABELS.png', recursive=True):
        img_paths.append(osp.join(gt_dir,img_path))

    if args.nproc > 1:
        track_parallel_progress(convert_labels, img_paths, args.nproc)
    else:
        track_progress(convert_labels, img_paths)


if __name__=='__main__':
    main()
