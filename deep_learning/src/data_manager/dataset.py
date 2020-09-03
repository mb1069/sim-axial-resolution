import os

import numpy as np
import glob

from torch.utils.data import Dataset
from tifffile import imread


# Wrapper class for torch Dataset instance

def is_tif(fname):
    return '.tif' in fname


def is_npy(fname):
    return '.npy' in fname


class DataSetManager(Dataset):
    def __init__(self, dirname, preload=True, dims=3):
        # Dimensionality of data, in format C Z X Y or C X Y
        self.dims = dims
        self.dirname = dirname
        self.file_list = list(filter(is_tif, glob.glob(os.path.join(dirname, '*'))))
        self.file_pairs = pair_files(self.file_list)
        self.file_pairs = sorted(self.file_pairs, key=lambda x: x[0])
        self.preloaded = preload
        if preload:
            self.data = list(self.preload_points())
        else:
            self.data = []

    def __getitem__(self, i):
        if self.preloaded:
            return self.data[i]
        else:
            raw, processed = self.file_pairs[i]

            return self.filepair2data(raw, processed)

    def preload_points(self):
        print('Preloading data')
        for raw, processed in self.file_pairs:
            yield self.filepair2data(raw, processed)
        print('Finished pre-loading data')

    def filepair2data(self, raw, processed):
        raw_data = imread(raw)
        processed_data = imread(processed)
        if self.dims == 4 and len(raw_data.shape) == 3:
            raw_data = raw_data[np.newaxis, :, :, :]
            processed_data = processed_data[np.newaxis, :, :, :]
        return {
            'raw': raw_data,
            'raw_name': os.path.basename(raw),
            'processed': processed_data,
            'processed_name': os.path.basename(processed),
        }

    def __len__(self):
        return len(self.file_pairs)


def pair_files(filelist):
    pairs = []
    for f in filelist:
        if '_in' in f:
            f_out = f.replace('_in', '_out')
            if f_out in filelist:
                pairs.append((f, f_out))
    return pairs
