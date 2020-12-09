import glob
from tifffile import imread, imwrite
import numpy as np
# Small script to normalise all image_creation in dir to range [0,1]

if __name__ == '__main__':
    imgs = glob.glob('/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/raw_imgs/*.tif')
    for f in imgs:
        img_data = imread(f)
        img_data = img_data / img_data.max()
        img_data = img_data.astype(np.float)
        imwrite(f, img_data, compress=6)
