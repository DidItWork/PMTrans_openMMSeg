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

    mappings = [255,10,2,0,1,4,8,5,13,7,11,18,17,0,255,6,9,12,14,15,16,3,0]
    
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
