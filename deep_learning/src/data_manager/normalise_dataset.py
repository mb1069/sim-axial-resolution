import glob
import os
from tifffile import imread, imwrite
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

dirname = '/Volumes/Samsung_T5/all_zeiss_datapoints'
output_dirname = dirname + '_normalised'

os.makedirs(output_dirname, exist_ok=True)
datapoints = glob.glob(os.path.join(dirname, '*.tif'))
n_datapoints = len(datapoints)


def get_img_sum(img_name):
    img = imread(img_name)
    return img.mean()

#
# with Pool(8) as p:
#     total = sum(list(tqdm(p.imap_unordered(get_img_sum, datapoints), total=n_datapoints)))

mean = 8.106048194246352e-05


print('Mean value is ',  str(mean))


def subtract_mean(img_name):
    img = imread(img_name)
    img -= mean
    img[img < 0] = 0
    outpath = os.path.join(output_dirname, os.path.basename(img_name))
    img = img.astype(np.float32)

    imwrite(outpath, img, compress=6)


with Pool(8) as p:
    list(tqdm(p.imap_unordered(subtract_mean, datapoints), total=n_datapoints))
