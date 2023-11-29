import argparse
import os.path as osp
import numpy as np
from imageio.v2 import imread, imwrite
# from mmcv import imread, imwrite

from mmengine.utils import scandir

def parse_args():
    parser = argparse.ArgumentParser(
        description='Formating raw SYNTHIA_RAND_CITYSCAPES')
    parser.add_argument('synthia_GT_path', help='synthia ground truth data path')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    gt_dir = args.synthia_GT_path

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
    
    for img_path in scandir(gt_dir, '_trainLabels.png', recursive=True):
        print(f'Converting {img_path}')
        synthia_img = np.asarray(imread(osp.join(gt_dir,img_path)),np.uint8)
        # print(synthia_img)
        cityscape_img = np.zeros(synthia_img.shape,np.uint8)
        for row in range(synthia_img.shape[0]):
                for col in range(synthia_img.shape[1]):
                    cityscape_img[row][col] = mappings[synthia_img[row][col]]
        # print(cityscape_img)    

        imwrite(osp.join(gt_dir,img_path.replace('_trainLabels','_cityscapesLabels')),cityscape_img)
    


if __name__=='__main__':
    main()
