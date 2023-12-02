from mmseg.apis import init_model, inference_model
from mmengine.utils import (mkdir_or_exist, scandir)
from imageio.v2 import imwrite
import os.path as osp
import argparse
from numpy import asarray
import numpy as np
from math import ceil

def label(config_path=None, checkpoint_path=None, img_dir=None, out_dir=None) -> None:

    if config_path is None:
        config_path = 'configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x512.py'
    
    if checkpoint_path is None:
        checkpoint_path = 'work_dirs/segformer_mit-b0_8xb1-160k_synthia-512x512/iter_32000.pth'
    
    assert img_dir is not None, 'please provide directory for images to label'

    if out_dir==None:
        out_dir = img_dir

    print("Loading model")

    # init model and load checkpoint
    model = init_model(config_path, checkpoint_path)

    img_paths = []

    imgs = []

    batch_size = 10

    for path in scandir(img_dir, '.png', recursive=True):

        img_paths.append(osp.join(img_dir,path))

        imgs.append(path)

    itrs = ceil(len(img_paths)/batch_size)

    for i in range(itrs):

        if i < itrs-1:
            image_paths = img_paths[i*batch_size:(i+1)*batch_size]
            images = imgs[i*batch_size:(i+1)*batch_size]
        else:
            image_paths = img_paths[i*batch_size:]
            images = imgs[i*batch_size:]

        print(f"Starting inference, batch {i} of {itrs}")

        results = inference_model(model, image_paths)

        for result,img_path in zip(results,images):

            print("Converting", img_path)

            label_img = result.pred_sem_seg.data.squeeze(0).cpu()

            label_img = asarray(label_img,np.uint8)

            path = osp.join(out_dir,img_path.replace('.png','_cityscapesLabels.png'))

            mkdir_or_exist('/'.join(path.split('/')[:-1]))

            imwrite(path,label_img)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generating labels for dataset')
    parser.add_argument('--config_path', help='path of model configuration file',type=str)
    parser.add_argument('--chkpt_path', help='path of model checkpoint', type=str)
    parser.add_argument('--img_dir', help='directory of images to label', type=str)
    parser.add_argument('--out_dir', help='directory of output images', type=str)
    args = parser.parse_args()
    return args

if __name__=='__main__':

    args = parse_args()

    config_path = args.config_path
    checkpoint_path = args.chkpt_path
    img_dir = args.img_dir
    out_dir = args.out_dir

    label(config_path, checkpoint_path, img_dir, out_dir)