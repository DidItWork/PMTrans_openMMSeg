import mmcv
import imageio
import numpy as np

def main():
    imageio.plugins.freeimage.download()
    cityscape_path = 'data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png'
    synthia_path = 'data/synthia/GT/train/0000001_trainLabels.png'

    # cityscape_img = mmcv.imread(cityscape_path)
    synthia_img = np.asarray(imageio.v2.imread(synthia_path))
    # synthia_img = np.asarray(imageio.v2.imread(synthia_path, format="PNG-FI"),np.uint8)
    # synthia_img[:,:,1] = synthia_img[:,:,0]
    # synthia_img[:,:,2] = synthia_img[:,:,0]
    # # for row in range(synthia_img.shape[0]):
    # #     for col in range(synthia_img.shape[1]):
    # #         synthia_img[row][col] = [255,255,255]

    # print(cityscape_img)
    # print(np.unique(synthia_img[:,:,0]))
    # print(np.unique(synthia_img[:,:,1]))
    # print(np.unique(synthia_img[:,:,2]))
    # # print(type(synthia_img[0][0][0]))
    print(synthia_img)
    # imageio.v2.imwrite(synthia_path.replace('LABELS','trainLabels'), synthia_img[:,:,0])

if __name__=='__main__':
    main()