import time
import numpy as np
import os
from torch.nn import Conv3d
from tifffile import imread
import torch
import glob

# Script used to evaluate execution time of deep learning models using Nvidia CUDA GPUs
f_dir_name = os.path.dirname(__file__)
model_file = os.path.join(f_dir_name, 'models', 'MSE_2D_RCAN_w_3D_Conv_300_epochs_3chunk.pth')

images_dir = os.path.join(f_dir_name, 'raw_imgs')

im_name = '7_0_8000points_in.tif'

im_path = os.path.join(images_dir, im_name)


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


block_size = 13

def run_model(model, inp_files, n_iter):
    first_conv = get_model_first_convolution(model)
    in_channels = get_n_in_channels(first_conv)

    count = 0

    for inp_file in inp_files:
        print(inp_file)
        for i in range(n_iter):
            in_data = imread(inp_file)

            indices = list(range(0, in_data.shape[0], in_channels))[1:]
            chunked_in_data = np.split(in_data, indices)
            chunked_out_data = []
            while len(chunked_in_data):
                block = np.vstack(chunked_in_data[0:block_size])
                chunked_in_data = chunked_in_data[block_size:]
                try:
                    block_data = block.reshape(block_size, 1, 21, 256, 256)
                except ValueError:
                    break
                block_data = torch.from_numpy(block_data).float()
                with torch.no_grad():
                    chunk_out = model(block_data).cpu().numpy()
                chunked_out_data.append(chunk_out)
            torch.cuda.empty_cache()
            count += 1
            print(count)

if __name__ == '__main__':
    model = torch.load(model_file)
    model.float()
    torch.cuda.reset_peak_memory_stats()
    if torch.cuda.device_count() == 1:
        model = list(model.children())[0]

    model.eval()
    print(f'Num devices: {torch.cuda.device_count()}')
    n_iter = 10

    inp_files = glob.glob(os.path.join(images_dir, '*.tif'))
    print(f'{len(inp_files)} images')


    t1 = time.time()
    im_out = run_model(model, inp_files, n_iter)
    t2 = time.time()

    print(f'Time taken: {round((t2 - t1)/(n_iter*len(inp_files)), 2)} secs.')

    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.peak"]
    print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")