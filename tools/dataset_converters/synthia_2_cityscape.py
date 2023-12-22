import argparse
import os.path as osp
import numpy as np
from imageio.v2 import imread, imwrite
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)
# from mmcv import imread, imwrite

from shutil import move
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(
        description='Formating raw SYNTHIA_RAND_CITYSCAPES labels')
    parser.add_argument('synthia_path', help='synthia data path')
    parser.add_argument('--train-split', help='training split 0.0 to 1.0')
    parser.add_argument('--val-split', help='validation split 0.0 to 1.0')
    parser.add_argument('--test-split', help='testing split 0.0 to 1.0')
    parser.add_argument('--gt-dir', default='GT', type=str)
    parser.add_argument('--img-dir', default='RGB', type=str)
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args

def convert_labels(img_path, mappings):
    synthia_img = np.asarray(imread(img_path, format="PNG-FI"),np.uint8)[:,:,0]
    cityscape_img = mappings[synthia_img] 
    imwrite(img_path.replace('.png','_cityscapesLabels.png'),cityscape_img)

def split_tvt(img_path, gt_dir, labels_dir, img_dir, split):

    move(osp.join(img_dir,img_path),osp.join(img_dir,split,img_path))
    move(osp.join(labels_dir,img_path.replace(".png","_cityscapesLabels.png")),osp.join(gt_dir,split,img_path.replace(".png","_cityscapesLabels.png")))


def main():

    mappings = np.array([255, #0
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
                ],dtype=np.uint8)

    args = parse_args()

    gt_dir = osp.join(args.synthia_path, args.gt_dir)

    labels_dir = osp.join(gt_dir, "LABELS")

    img_dir = osp.join(args.synthia_path, args.img_dir)

    train_split = args.train_split if args.train_split else 0.6
    val_split = args.val_split if args.val_split else 0.2
    test_split = args.test_split if args.test_split else 1-train_split-val_split

    assert(train_split>0 and val_split>0 and test_split>0 and train_split+val_split+test_split==1)

    img_paths = []
    for img_path in scandir(img_dir, '.png', recursive=True):
        img_paths.append(osp.join(labels_dir,img_path))

    print("Converting labels")

    if args.nproc > 1:
        track_parallel_progress(partial(convert_labels, mappings=mappings),
                                img_paths, args.nproc)
    else:
        track_progress(partial(convert_labels, mappings=mappings),
                       img_paths)
    
    print("train:", train_split)
    print("val:", val_split)
    print("test", test_split)

        
    images_paths = list(scandir(img_dir,".png", recursive=True))

    train_split=int(train_split*len(images_paths))

    val_split=int(val_split*len(images_paths))

    test_split=len(images_paths)-train_split-val_split

    splits = {"train" : train_split, "val" : val_split, "test" : test_split}

    start = 0

    for split,val in splits.items():
        mkdir_or_exist(osp.join(gt_dir,split))
        mkdir_or_exist(osp.join(img_dir,split))

        print(f"Splitting {split} dataset")
    
        if args.nproc > 1:
            track_parallel_progress(partial(split_tvt, gt_dir = gt_dir, labels_dir = labels_dir, img_dir = img_dir, split = split),
                                    images_paths[start:start+val], args.nproc)
        else:
            track_progress(partial(split_tvt, gt_dir = gt_dir, labels_dir = labels_dir, img_dir = img_dir, split = split),
                           images_paths[start:start+val])
        
        with open(osp.join(args.synthia_path, f'{split}.txt'), 'w') as f:
            f.writelines(images_paths[i].replace('.png','')+ '\n' for i in range(start, start+val))
        
        start+=val


if __name__=='__main__':
    main()
