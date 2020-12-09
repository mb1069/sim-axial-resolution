import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as mp
from scipy import fftpack
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity as ssim
from tifffile import imread, imwrite
from torch.nn import Conv2d, Conv3d
from torch.utils.data import Dataset, DataLoader

# Script to generate all results csv used to generate R charts
# Results from RCAN models are generated on the fly if no file matching the expected pattern is found (<model_name>_out.tif)
show_charts = False

cuda_avail = torch.cuda.is_available()

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

num_input_raw_frames = 7

pixel_sizes = {
    'XY': 54.2,
    'X': 54.2,
    'Y': 54.2,
    'Z': 116.66,
}


def norm_img(img):
    img[img < 0] = 0
    return img / img.max()


def mse(A, B):
    if A.shape[0] != B.shape[0]:
        max_frames = min(A.shape[0], B.shape[0])
        A = A[0:max_frames]
        B = B[0:max_frames]
    return ((A - B) ** 2).mean(axis=None)


def peak(x, c):
    return np.exp(-np.power(x - c, 2) / 16.0)


def lin_interp(x, y, i, half):
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def half_max_x(x, y):
    half = max(y) / 2
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = list(np.where(zero_crossings)[0])
    lhs_crossing = list(filter(lambda c: c < len(y) // 2, zero_crossings_i))[-1]

    rhs_crossing = zero_crossings_i[zero_crossings_i.index(lhs_crossing) + 1]

    hmx = [lin_interp(x, y, lhs_crossing, half),
           lin_interp(x, y, rhs_crossing, half)]

    return hmx


def load_norm_img(fpath):
    data = imread(fpath)

    data = data / data.max()
    return data


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def gauss_pairs(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return gauss(x, a1, mu1, sigma1) + gauss(x, a2, mu2, sigma2)


def fit_curve(x, y):
    max_val = np.real(y.max())
    mu_val = x.mean()
    # Autcorrelation function
    func1_min_bounds = [max_val / 50, mu_val - 0.0000001, 0.01]
    func1_max_bounds = [max_val, mu_val + 0.0000001, np.sqrt(x.max() / 2)]
    func1_estimate = [max_val / 3, mu_val, 1]

    # Interference function
    func2_min_bounds = [0, mu_val - 0.000000, 0]
    func2_max_bounds = [max_val, mu_val + 0.0000001, np.inf]
    func2_estimate = [max_val / 1.5, mu_val, 10]

    bounds = (func1_min_bounds + func2_min_bounds, func1_max_bounds + func2_max_bounds)
    p0 = func1_estimate + func2_estimate

    popt, pcov = curve_fit(gauss_pairs, x, y, p0=p0, bounds=bounds)
    return popt[0:3], popt[3:]


def get_2nd_fwhm(series, series_len, img_name, model_name):
    x = np.linspace(-series_len / 2, (series_len / 2) - 1, len(series))

    param1, param2 = fit_curve(x, series)

    y1 = gauss(x, *param1)
    y2 = gauss(x, *param2)
    if show_charts or True:
        l2, = mp.plot(x, y1, label='Gauss1', linestyle='--')
        l3, = mp.plot(x, y2, label='Gauss2', linestyle='-.')
        lfull, = mp.plot(x, y1 + y2, label='Gauss1+Gauss2', linestyle=':')

        lreal, = mp.plot(x, np.real(series), label='Real')
    # Select gaussian with smallest standard dev unless one has insignificant amplitude
    if param1[2] < param2[2] and param1[0] > 1:
        params = param1
    else:
        params = param2
    series = gauss(x, *params)
    try:
        hmx = half_max_x(x, series)
        fwhm = (hmx[1] - hmx[0])
        half = float(max(np.real(series)) / 2)
        if show_charts:
            mp.plot(np.real(hmx), [half, half])
    except IndexError:
        print('No axis crossing')
    if show_charts or True:
        mp.legend(handles=[l2, l3, lfull, lreal])
        mp.title(f'{model_name}')
        mp.show()
    return fwhm


#
# def fwhm(series, series_len, img_name, model_name):
#     x = np.linspace(-series_len/2, (series_len/2)-1, len(series))
#
#     param1, param2 = fit_curve(x, series)
#     fwhm2 = get_2nd_fwhm(x, param1, param2, img_name, model_name)
#     # y1 = gauss(x, *param1)
#     # y2 = gauss(x, *param2)
#     # if show_charts:
#     #     l2, = mp.plot(x, y1, label='Gauss1', linestyle='--')
#     #     l3, = mp.plot(x, y2, label='Gauss2', linestyle='-.')
#     #     lfull, = mp.plot(x, y1 + y2, label='Gauss1+Gauss2', linestyle=':')
#     #
#     #     lreal, = mp.plot(x, np.real(series), label='Real')
#     # if show_charts:
#     #     if 'RCAN' in model_name:
#     #         model_name = 'RCAN_' + model_name.split('_')[-1]
#     #     mp.title(f'{img_name}_{model_name}')
#     # if show_charts:
#     #     mp.legend(handles=[lreal, l2, l3, lfull])
#     #     mp.show(block=False)
#
#     return fwhm2


def benchmark_fwhm(img, img_name, model_name):
    print(img.shape)
    fft = fftpack.fftn(img)

    pspectrum = np.abs(fft) ** 2

    results = []

    for axis, c in zip(('Z', 'XY'), [(1, 2), (0,)]):

        print(f'\tProcessing axis {axis}')
        summed_pspectrum = np.sum(pspectrum, axis=c)
        if axis == 'XY':
            series_len = len(summed_pspectrum)
            half_len = series_len // 2
            inv_fft_spectrum = np.concatenate(
                [summed_pspectrum[0:half_len], np.zeros((10000, summed_pspectrum.shape[0])),
                 summed_pspectrum[half_len:]])

            # Peak method
            inv_fft_spectrum = fftpack.ifft2(inv_fft_spectrum)
            inv_fft_spectrum = fftpack.fftshift(inv_fft_spectrum)
            max_val = inv_fft_spectrum.max()
            location = np.where(inv_fft_spectrum == max_val)
            inv_fft_spectrum = inv_fft_spectrum[:, location[1][0]]
        else:
            series_len = len(summed_pspectrum)
            half_len = series_len // 2

            summed_pspectrum = np.concatenate(
                [summed_pspectrum[0:half_len], np.zeros(10000), summed_pspectrum[half_len:]])

            inv_fft_spectrum = fftpack.ifft(summed_pspectrum)
            inv_fft_spectrum = fftpack.fftshift(inv_fft_spectrum)

        fwhm_val = np.real(get_2nd_fwhm(inv_fft_spectrum, series_len, img_name, f'{model_name}_{axis}')) / np.sqrt(2)
        results.append((axis, fwhm_val))

    return dict(results)


def get_ssim(img_data, reference_out_img, win_size=11):
    s1 = img_data.shape[0]
    s2 = reference_out_img.shape[0]

    if s1 != s2:
        sm = min(s1, s2)
        img_data = img_data[0:sm]
        reference_out_img = reference_out_img[0:sm]
    return ssim(img_data, reference_out_img, win_size=win_size)


def get_model_first_convolution(model):
    m = model
    while not (isinstance(m, Conv2d) or isinstance(m, Conv3d)):
        m = list(m.children())[0]
    return m


def get_n_in_channels(first_layer):
    if isinstance(first_layer, Conv2d):
        return first_layer.in_channels
    elif isinstance(first_layer, Conv3d):
        return first_layer.kernel_size[0]
    else:
        raise EnvironmentError('Unknown first layer type.')


def get_model_results_filename(model_file, inp_file, results_dir):
    model_basename = os.path.splitext(os.path.basename(model_file))[0]

    out_file = os.path.splitext(os.path.basename(inp_file))[0]
    out_file = os.path.join(results_dir, out_file.replace('_in', '_out') + '_' + model_basename + '.tif')
    return out_file


def load_model(model_file):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print('setting model on cuda')
        model = torch.load(model_file, map_location=device)
    else:
        model = torch.load(model_file, map_location=device)

    # Unpack from dataParallel
    if len(list(model.children())) == 1:
        model = list(model.children())[0]

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    model = model.to(device)
    return model


class EvalDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len




def reconstruct_imgs(model, in_data_stacked):
    num_images = in_data_stacked.shape[0]

    # Chunk all image_creation first

    first_conv = get_model_first_convolution(model)
    in_channels = get_n_in_channels(first_conv)
    num_chunks = int(in_data_stacked.shape[1] / in_channels)
    slices = num_chunks * in_channels
    in_data = in_data_stacked[:, :slices, :, :]

    im_chunks = []
    for image in in_data:
        chunks = np.split(image, num_chunks)
        im_chunks.extend(chunks)

        # Stack all chunks together
    in_data = np.stack(im_chunks)

    # Insert empty channel dimension
    in_data = in_data[:, np.newaxis, :, :, :]

    # Adjust depending on GPU memory
    max_parallel_load = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EvalDataset(in_data)

    dataloader = DataLoader(dataset=dataset, batch_size=max_parallel_load, shuffle=False, num_workers=8)

    out_datas = []

    processed = 0
    for data in dataloader:
        chunk_data = data.to(device, dtype=torch.float32)

        with torch.no_grad():
            output = model(chunk_data)
        data_out = np.concatenate(output.cpu().numpy(), axis=0)

        out_datas.append(data_out)
        processed += len(data)

    # num_chunks = in_data.shape[0] // max_parallel_load
    # in_datas = np.array_split(in_data, num_chunks, axis=0)
    # out_datas = []
    # i = 1
    # for in_data in in_datas:
    #     print(4, i, time.time() - t1)
    #     i += 1
    #     in_data = torch.from_numpy(in_data).float()
    #     if cuda_avail:
    #         in_data = in_data.to('cuda')
    #
    #     with torch.no_grad():
    #         data_out = model(in_data)
    #     data_out = np.concatenate(data_out.cpu().numpy(), axis=0)
    #
    #     out_datas.append(data_out)
    #
    # print(5, time.time() - t1)
    data_out = np.concatenate(out_datas, axis=0)

    reconstructed_images = np.split(data_out, num_images, axis=0)

    return reconstructed_images


def reconstruct_img(model, in_data):
    first_conv = get_model_first_convolution(model)
    in_channels = get_n_in_channels(first_conv)

    # Full stack loading
    num_chunks = int(in_data.shape[0] / in_channels)
    slices = num_chunks * in_channels
    in_data = in_data[:slices, :, :]

    # Truncate stack to multiple of chunk size
    in_data = np.split(in_data, num_chunks, axis=0)
    in_data = [i[np.newaxis, :, :, :] for i in in_data]
    in_data = np.concatenate(in_data, axis=0)

    in_data = in_data[:, np.newaxis, :, :, :]

    in_data = torch.from_numpy(in_data).float()
    if cuda_avail:
        in_data = in_data.to('cuda')

    with torch.no_grad():
        data_out = model(in_data)
        data_out = data_out.cpu().numpy()
    data_out = np.concatenate(data_out, axis=0)

    # Chunk-wise stack loading
    # chunks = int(in_data.shape[0] / in_channels)
    # chunks_data = []
    # for c in trange(chunks):
    #     chunk_data = in_data[c * in_channels:(c + 1) * in_channels]
    #
    #     if isinstance(first_conv, Conv3d):
    #         chunk_data = chunk_data[np.newaxis, np.newaxis, :, :, :]
    #     else:
    #         chunk_data = chunk_data[np.newaxis, :, :, :]
    #     chunk_data = torch.from_numpy(chunk_data).float()
    #     with torch.no_grad():
    #         chunk_out = model(chunk_data)
    #     chunks_data.append(chunk_out.cpu().numpy()[0])

    # Center-chunk-wise stack loading
    # num_chunks = int(in_data.shape[0] / 7)
    # chunks_data = []
    #
    # for c in range(num_chunks - 2):
    #     input_start = c * 7
    #     input_end = (c * 7) + in_channels
    #     if input_end > in_data.shape[0]:
    #         break
    #     chunk_data = in_data[input_start:input_end]
    #     print('IN', chunk_data.shape, input_start, input_end)
    #
    #     if isinstance(first_conv, Conv3d):
    #         chunk_data = chunk_data[np.newaxis, np.newaxis, :, :, :]
    #     else:
    #         chunk_data = chunk_data[np.newaxis, :, :, :]
    #     chunk_data = torch.from_numpy(chunk_data).float()
    #     with torch.no_grad():
    #         chunk_out = model(chunk_data)
    #
    #     if c == 0:
    #         start = 0
    #     else:
    #         start = 3
    #
    #     if c == num_chunks - 2:
    #         end = 9
    #     else:
    #         end = 6
    #
    #     chunk_out = chunk_out.cpu().numpy()[0][start:end]
    #     chunks_data.append(chunk_out)
    #     print('OUT', chunk_out.shape)
    # data_out = np.concatenate(chunks_data, axis=0)

    # num_chunks = int(in_data.shape[0] / 7)
    # chunks_data = []
    #

    return data_out


def run_model_wrapper(model, model_file, inp_file, results_dir):
    out_file = get_model_results_filename(model_file, inp_file, results_dir)
    if os.path.exists(out_file):
        print(f'Result exists: {out_file}')
        return imread(out_file)
    print(out_file)

    in_data = imread(inp_file).squeeze()
    # Normalise image
    in_data = in_data.astype(np.float)
    in_data = in_data / in_data.max()

    data_out = reconstruct_img(model, in_data)

    print(out_file)
    imwrite(out_file, data_out, compress=6)

    return data_out


class ResultGenerator:
    def __init__(self, models, raw_img_dir, out_fname, current_dir, processed_imgs_dir):
        self.current_dir = current_dir
        self.processed_imgs_dir = processed_imgs_dir
        self.models = models
        self.raw_img_dir = raw_img_dir
        self.out_fname = out_fname
        self.all_results = []

        self.run_all_benchmarks()
        self.write_csv()

    def run_all_benchmarks(self):
        self.all_results = []
        ref_imgs_in = glob.glob(os.path.join(self.current_dir, self.raw_img_dir, '*_in*.tif'))
        for model_file in self.models:
            if model_file not in non_model_comparisons:
                model = load_model(model_file)
                if cuda_avail:
                    model = model.to('cuda')
            for img_in_name in ref_imgs_in:
                print(f'Processing {os.path.basename(model_file)}, {os.path.basename(img_in_name)}')
                reference_out_img = imread(img_in_name.replace('_in', '_out'))
                if model_file == 'reference':
                    img_data = reference_out_img
                elif model_file == 'sim':
                    sim_img_name = os.path.basename(img_in_name.replace('_in.tif', '_sim.tif'))
                    sim_img_name = os.path.join(self.processed_imgs_dir, sim_img_name)
                    if sim_img_name == img_in_name:
                        raise EnvironmentError(f'Could not find a sim img for {sim_img_name}')
                    img_data = imread(sim_img_name)
                else:
                    img_data = run_model_wrapper(model, model_file, img_in_name, self.processed_imgs_dir)

                img_data = norm_img(img_data)
                reference_out_img = norm_img(reference_out_img)

                model_result = benchmark_fwhm(img_data, os.path.basename(img_in_name), os.path.basename(model_file))
                for axis in [k for k in pixel_sizes.keys()]:
                    model_result[axis] *= pixel_sizes[axis]
                model_result['mse'] = float(mse(img_data, reference_out_img))
                model_result['ssim'] = get_ssim(img_data, reference_out_img, win_size=3)
                model_result['img'] = os.path.basename(img_in_name)
                model_result['model'] = os.path.basename(model_file)
                self.all_results.append(model_result)

    def write_csv(self):
        df2 = pd.DataFrame.from_dict(self.all_results)
        df2 = df2.sort_values(['img', 'model'])
        df2.to_csv(self.out_fname)
        print(df2)
        print(self.out_fname)


def get_masked_layer(img_name):
    m = re.findall(r'\d+_struct_(.+)_mask_in.tif', img_name)
    if len(m):
        return m[0]
    return 'none'


class MaskedResultGenerator(ResultGenerator):
    def write_csv(self):
        df2 = pd.DataFrame.from_dict(self.all_results)
        df2['masked_layer'] = df2['img'].map(get_masked_layer)
        df2 = df2.sort_values(['img', 'model'])
        print(df2)
        df2.to_csv(self.out_fname)


if __name__ == '__main__':
    non_model_comparisons = ['reference', 'sim']
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--results-dir',
                        default='/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/')
    args = parser.parse_args()

    current_dir = args.results_dir
    processed_images = os.path.join(current_dir, 'processed_images')
    models_dir = os.path.join(current_dir, 'models')

    all_models = glob.glob(
        os.path.join(models_dir, '*.pth')) + non_model_comparisons
    ResultGenerator(all_models, 'raw_imgs', 'all_results.csv', current_dir, processed_images)

    # final_models_no_ref = glob.glob(os.path.join(models_dir, 'MSE_2D_RCAN_w_3D_Conv_300_epochs_3chunk.pth')) + ['sim']
    # ResultGenerator(final_models_no_ref, 'noisy_imgs', 'noise_results.csv', current_dir, results_dir)
    #
    # masked_models = glob.glob(os.path.join(models_dir, 'MSE_2D_RCAN_w_3D_Conv_300_epochs_3chunk.pth'))
    # MaskedResultGenerator(masked_models, 'masked_raw_imgs', 'masked_layers.csv', current_dir, results_dir)
    #
    # final_models_w_ref = glob.glob(
    #     os.path.join(models_dir, 'MSE_2D_RCAN_w_3D_Conv_300_epochs_3chunk.pth')) + non_model_comparisons
    # ResultGenerator(final_models_w_ref, 'external_images', 'external.csv', current_dir, results_dir)
