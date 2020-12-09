import argparse
import os
from tifffile import imread, imwrite
import glob
import numpy as np
from image_simulation.HexSimProcessor.HexSimProcessor import HexSimProcessor
from tqdm import tqdm


def get_outpath(out_dir, inpath):
    return os.path.join(out_dir, os.path.basename(inpath).replace('_in', '_sim'))


def run_fairsim(impath):

    # Calibration step
    imdata = imread(impath)
    imdata = np.single(imdata)
    # Fake calibration with exact parameters
    h = HexSimProcessor()
    h.fake_calibrate(imdata)

    # Reconstruct img
    out_img = h.batchreconstruct(imdata, n_output_frames=120)

    # Normalise and cast to float32
    out_img = (out_img / out_img.max()).astype(np.float32)
    return out_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir',
                        default='/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/tmp_imgs')
    args = parser.parse_args()

    raw_dir = os.path.join(args.dir, 'raw_imgs')
    out_dir = os.path.join(args.dir, 'processed_images')

    raw_imgs = glob.glob(os.path.join(raw_dir, '*_in.tif'))
    for impath in tqdm(raw_imgs):
        outpath = get_outpath(out_dir, impath)
        if os.path.exists(outpath):
            print(f'File exists: {outpath}')
        else:
            imdata = run_fairsim(impath)
            print(outpath)
            imwrite(outpath, imdata, compress=6)
