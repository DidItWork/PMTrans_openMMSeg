# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from os import listdir
from shutil import move

# from cityscapesscripts.preparation.json2labelImg import json2labelImg
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)


# def convert_json_to_label(json_file):
#     label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
#     json2labelImg(json_file, label_file, 'trainIds')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Formating raw SYNTHIA_RAND_CITYSCAPES')
    parser.add_argument('synthia_path', help='synthia data path')
    parser.add_argument('--gt-dir', default='GT', type=str)
    parser.add_argument('--img-dir', default='RGB', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--train-split', help='training split 0.0 to 1.0')
    parser.add_argument('--val-split', help='validation split 0.0 to 1.0')
    parser.add_argument('--test-split', help='testing split 0.0 to 1.0')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    synthia_path = args.synthia_path
    out_dir = args.out_dir if args.out_dir else synthia_path
    train_split = args.train_split if args.train_split else 0.6
    val_split = args.val_split if args.val_split else 0.2
    test_split = args.test_split if args.test_split else 1-train_split-val_split

    assert(train_split>0 and val_split>0 and test_split>0 and train_split+val_split+test_split==1)

    mkdir_or_exist(out_dir)

    gt_dir = osp.join(synthia_path, args.gt_dir)
    img_dir = osp.join(synthia_path, args.img_dir)

    # if args.nproc > 1:
    #     track_parallel_progress(convert_json_to_label, poly_files, args.nproc)
    # else:
    #     track_progress(convert_json_to_label, poly_files)

    gt_subs = listdir(gt_dir)

    print(gt_subs)

    img_files = []

    for png in scandir(img_dir, '.png', recursive=True):
        # img_file = osp.join(img_dir, png)
        img_files.append(png)
    
    print(img_files)

    train_split*=len(img_files)

    train_split = int(train_split)

    val_split*=len(img_files)

    val_split = int(val_split)

    test_split=len(img_files)-train_split-val_split

    print(train_split, val_split, test_split)

    # test_split = int(test_split)

    splits = {"train" : train_split, "val" : val_split, "test" : test_split}

    start = 0

         
    for split,val in splits.items():
        mkdir_or_exist(osp.join(gt_dir,split))
        mkdir_or_exist(osp.join(img_dir,split))
        # for gt_sub in gt_subs:
        #     mkdir_or_exist(osp.join(gt_dir,split,gt_sub))

        # for img_sub in img_subs:
        #     mkdir_or_exist(osp.join(img_dir,split,img_sub))
        
        for i in range(start, start+val):
            print(i)
            move(osp.join(img_dir,img_files[i]),osp.join(img_dir,split,img_files[i]))
            for sub_dir in gt_subs:
                move(osp.join(gt_dir,sub_dir,img_files[i]),osp.join(gt_dir,split,img_files[i].replace('.png','_'+sub_dir+'.png')))

        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(img_files[i].replace('.png','')+ '\n' for i in range(start, start+val))

        start += val


    # for split in splits.items:
    #     filenames = []
    #     for poly in scandir(
    #             osp.join(gt_dir, split), '_polygons.json', recursive=True):
    #         filenames.append(poly.replace('_gtFine_polygons.json', ''))
    #     with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
    #         f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()
