import numpy as np
import os
from torch.nn import Conv3d
from tifffile import imread
import torch
import glob

# Script used to evaluate execution time of deep learning models using Nvidia CUDA GPUs
from model_eval.src.evaluation.compare import reconstruct_img, reconstruct_imgs, load_model

f_dir_name = os.path.join(os.path.dirname(__file__), os.pardir, 'new_analysis')
model_file = os.path.join(f_dir_name, 'models', 'final.pth')

images_dir = os.path.join(f_dir_name, 'raw_imgs')


def get_n_in_channels(first_layer):
    if isinstance(first_layer, Conv3d):
        return first_layer.kernel_size[0]
    else:
        raise EnvironmentError('Unknown first layer type.')


def get_model_first_convolution(model):
    m = model
    while not isinstance(m, Conv3d):
        m = list(m.children())[0]
    return m

import time

def run_model2(model, inp_files, n_iter):
    count = 0

    for i in range(n_iter):
        t1 = time.time()
        in_data = np.stack([imread(i) for i in inp_files])
        in_data = in_data / 255

        img = reconstruct_imgs(model, in_data)
        torch.cuda.empty_cache()
        count += len(inp_files)
        print(count, time.time() - t1)


def run_model(model, inp_files, n_iter):
    count = 0
    print(inp_files)
    for inp_file in inp_files:
        print(inp_file)
        for i in range(n_iter):
            in_data = imread(inp_file)
            in_data = in_data / 255

            img = reconstruct_img(model, in_data)
            torch.cuda.empty_cache()
            count += 1
            print(count)


if __name__ == '__main__':
    model = load_model(model_file)
    if torch.cuda.is_available():
        print(f'Num devices: {torch.cuda.device_count()}')

    model.eval()
    n_iter = 10

    inp_files = glob.glob(os.path.join(images_dir, 'points*_in.tif'))
    print(f'{len(inp_files)} images')

    t1 = time.time()
    im_out = run_model(model, inp_files, n_iter)
    t2 = time.time()

    print(f'Time taken: {round((t2 - t1)/(n_iter*len(inp_files)), 2)} secs.')
