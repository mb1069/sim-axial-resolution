import numpy as np
from tifffile import imread
from matplotlib import pyplot as mp
from scipy import fftpack

from src.model_runner.benchmark.compare import half_max_x


def fwhm(series):
    # make some fake data

    # fwhms = []
    # for img_x in range(0, data.shape[1]):
    #     for img_y in range(0, data.shape[2]):
    #         series = data[:, img_x, img_y]
    #         if series.max() > 0.5:
    #             hmx = half_max_x(x, series)
    #             fwhm = hmx[1] - hmx[0]
    #             fwhms.append(fwhm)
    #             # break

    # print(np.mean(fwhms))
    # print(np.std(fwhms))
    # # find the two crossing points
    x = np.linspace(0, len(series), len(series))
    hmx = half_max_x(x, series)

    mp.plot(x, series)
    if hmx:
        half = max(series) / 2.0
        mp.plot(hmx, [half, half])

    mp.show()
    if hmx:
        fwhm = hmx[1] - hmx[0]
        return fwhm
    return None


img = imread('/Users/miguelboland/Projects/uni/project_3/src/model_runner/benchmark/processed_images/46_struct_out_MSE_2D_RCAN_w_3D_Conv_300_epoths_double.tif')
fft = fftpack.fftn(img)

pspectrum = abs(fft) ** 2

print(pspectrum.shape)

results = []

# Z, X, Y
for axis, c in zip(('X', 'Y', 'Z'), [(0, 1), (0, 2), (1, 2)]):
    summed_pspectrum = np.sum(pspectrum, axis=c)

    inv_fft_spectrum = fftpack.ifft(summed_pspectrum)

    inv_fft_spectrum = fftpack.fftshift(inv_fft_spectrum)

    results.append((axis, np.abs(fwhm(inv_fft_spectrum))))

for res in results:
    print(res)
