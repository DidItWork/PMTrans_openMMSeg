import argparse
import os.path as osp
import numpy as np
from imageio.v2 import imread, imwrite

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
    
    for img_path in scandir(gt_dir, 'LABELS.png', recursive=True):
        print(f'Converting {img_path}')
        synthia_img = np.asarray(imread(osp.join(gt_dir,img_path), format="PNG-FI"),np.uint8)
        imwrite(osp.join(gt_dir,img_path.replace('LABELS','trainLabels')),synthia_img[:,:,0])
    


if __name__=='__main__':
    main()
