import argparse
import os.path as osp
import numpy as np
from imageio.v2 import imread, imwrite
# from mmcv import imread, imwrite

from mmengine.utils import scandir

def parse_args():
    parser = argparse.ArgumentParser(
        description='Get mean and variance of RGB channels in images, reads images in H W C format')
    parser.add_argument('image_dir', help='directory of images')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    img_dir = args.image_dir

    imgs = []
    
    for img_path in scandir(img_dir, '.png', recursive=True):
        img = np.asarray(imread(osp.join(img_dir,img_path)),np.uint8)
        img = img.reshape((-1,img.shape[-1]))

        if len(imgs)<50:
            imgs.append(img)
        else:
            break

    imgs = np.array(imgs).reshape(-1,imgs[0].shape[-1])

    print(imgs.shape)

    mean = np.mean(imgs,axis=0)

    var = np.var(imgs,axis=0)

    print("Images mean:",mean)
    print("Images variance:",var)
    
if __name__=='__main__':
    main()
