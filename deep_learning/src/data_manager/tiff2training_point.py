import logging
import os
import argparse

import numpy as np
from tqdm import tqdm
from src.data_manager.dataset import pair_files
from tifffile import imread, imwrite
from multiprocessing import Pool


def normalise(img):
    # img_max = 255
    # img_min = 0
    # return np.nan_to_num((img - img_min) / (img_max - img_min))
    return img / 255
    # return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/Users/miguelboland/Projects/uni/tmp')
    parser.add_argument('--outdir', default='/Users/miguelboland/Projects/uni/tmp2')
    parser.add_argument('--nframes', default=1, type=int)
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


def resize(data):
    return data
    # return cv2.resize(data, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)


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


class NonOverlappingFramesSingleChannel(TiffImagePair):
    def __init__(self, in_path, out_path, n_channels):
        self.n_channels = n_channels
        self.in_path = in_path
        self.out_path = out_path
        self.in_img = imread(in_path)
        self.out_img = imread(out_path)

        self.z, self.x, self.y = self.in_img.shape
        logging.debug('Loaded ' + in_path)
        logging.debug('Loaded ' + out_path)

    # 3D rcan overlapping frames (ie input frames 1-9 -> output frames 1-3), channels are illuminations at each Z
    def __getitem__(self, i):
        in_img_data = np.array([self.in_img[i * n_frames * self.n_channels:(i + 1) * n_frames * self.n_channels]])
        # Re-order from Z, C, X, Y -> C, Z, X, Y
        out_img_data = np.array([self.out_img[i * n_frames * out_channels:(i + 1) * n_frames * out_channels]])

        return {'raw': normalise(in_img_data),
                'processed': normalise(out_img_data)}

    def __len__(self):
        return self.z // (n_frames * self.n_channels) - 1


class NonOverlappingFramesSingleChannel2DOutput(NonOverlappingFramesSingleChannel):

    # 3D rcan overlapping frames (ie input frames 1-9 -> output frames 1-3), channels are illuminations at each Z
    def __getitem__(self, i):
        data = super().__getitem__(i)
        data['processed'] = data['processed'].squeeze()
        return data


class Overlappingframes(TiffImagePair):
    # 2D RCAN3D, 3 overlapping frames (ie input frames 1-9 -> output frames 1-3), channels are illuminations across all Z
    # IE format is Z, X, Y
    def __getitem__(self, i):
        if i + in_channels + n_frames - 1 >= self.z:
            raise StopIteration
        in_img_data = np.array(self.in_img[i:i + in_channels + n_frames - 1])
        out_img_data = np.array(self.out_img[i + in_channels: i + in_channels + n_frames])
        return {'raw': normalise(in_img_data),
                'processed': normalise(out_img_data)}


class NonOverlappingFrames(TiffImagePair):
    # 2D RCAN3D, 3 consecutive frames (zeiss format w/ 3 ang * 5 phases)
    def __getitem__(self, i):
        start_in = i * in_channels
        end_in = (i + n_frames) * in_channels
        in_img_data = np.array(self.in_img[start_in:end_in])

        start_out = i * out_channels
        end_out = (i * out_channels) + output_frames
        out_img_data = np.array(self.out_img[start_out:end_out])
        return {'raw': normalise(in_img_data),
                'processed': normalise(out_img_data)}


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


# for p in pairs:
#     print(p)
#     convert_filepair(p)
#     quit()
with Pool(8) as p:
    res = list(tqdm(p.imap_unordered(convert_filepair, pairs), total=len(pairs)))
