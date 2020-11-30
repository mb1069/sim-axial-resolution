import logging
import os
import argparse

import numpy as np
from tqdm import tqdm
from deep_learning.src.data_manager.dataset import pair_files
from tifffile import imread, imwrite
from multiprocessing import Pool


# Script to transform directory of full SIM image stacks into training data
# Arguments:
#    input_dir: directory containing pairs of SIM image_creation and high-res image stacks in format <n_in.tif, n_lout.tif>
#    outdir: output_directory
#    nframes: number of SIM reconstructions to include per training point (i.e concurrent processing chunk size of network)

def normalise(img):
    return img / 255


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--nframes', default=3, type=int)
    return parser.parse_args()


args = parse_args()
dirname = args.input_dir
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

n_frames = int(args.nframes)
in_channels = 7
out_channels = 3
input_frames = n_frames * in_channels

output_frames = n_frames * out_channels

imgs = os.listdir(dirname)

pairs = pair_files(imgs)


class TiffImagePair:
    def __init__(self, in_path, out_path, n_channels):
        self.n_channels = n_channels
        self.in_path = in_path
        self.out_path = out_path
        self.in_img = imread(in_path)
        self.out_img = imread(out_path)

        self.z, self.x, self.y = self.in_img.shape
        logging.debug('Loaded ' + in_path)
        logging.debug('Loaded ' + out_path)

    def __getitem__(self, i):
        raise NotImplemented('Subclass not called')

    def __len__(self):
        return int(self.z / in_channels) - n_frames + 1

    def seek_frames(self, i):
        try:
            self.in_img.seek(i)
            self.out_img.seek(i)
        except TypeError as e:
            print('FAILED', self.z, i, self.in_path)
            raise e


class NonOverlappingFramesSingleChannel2DOutput(TiffImagePair):
    def __init__(self, in_path, out_path, n_channels):
        self.n_channels = n_channels
        self.in_path = in_path
        self.out_path = out_path
        self.in_img = imread(in_path)
        self.out_img = imread(out_path)

        self.z, self.x, self.y = self.in_img.shape
        logging.debug('Loaded ' + in_path)
        logging.debug('Loaded ' + out_path)

    def __getitem__(self, i):
        in_start = i * n_frames * self.n_channels
        in_end = (i + 1) * n_frames * self.n_channels
        in_img_data = np.array([self.in_img[in_start:in_end]])

        # Re-order from Z, C, X, Y -> C, Z, X, Y
        out_start = i * n_frames * out_channels
        out_end = (i + 1) * n_frames * out_channels
        out_img_data = np.array([self.out_img[out_start:out_end]])

        out_img_data = out_img_data.squeeze()

        return {'raw': normalise(in_img_data),
                'processed': normalise(out_img_data)}

    def __len__(self):
        return self.z // (n_frames * self.n_channels) - 1


def convert_filepair(pair):
    img_in, img_out = pair
    in_basename = os.path.splitext(img_in)[0]
    out_basename = os.path.splitext(img_out)[0]
    img_in, img_out = os.path.join(dirname, img_in), os.path.join(dirname, img_out)
    tpairs = NonOverlappingFramesSingleChannel2DOutput(img_in, img_out, in_channels)
    for i in range(len(tpairs)):
        datapoint = tpairs[i]
        raw = datapoint['raw']
        processed = datapoint['processed']

        in_fname = os.path.join(outdir, in_basename + '_f_' + str(i) + '.tif')
        out_fname = os.path.join(outdir, out_basename + '_f_' + str(i) + '.tif')

        if raw.mean() < 5e-07 or processed.mean() < 5e-07:
            continue
        raw = raw.astype(np.float32)
        processed = processed.astype(np.float32)
        imwrite(in_fname, raw, compress=6)
        imwrite(out_fname, processed, compress=6)


for p in pairs:
    convert_filepair(p)

# Uncomment to use multiple threads
# with Pool(8) as p:
#     res = list(tqdm(p.imap_unordered(convert_filepair, pairs), total=len(pairs)))
