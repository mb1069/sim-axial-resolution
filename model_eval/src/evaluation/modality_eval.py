import os
from tifffile import imread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

z_pixel_nm = 116.66


def avg(lst):
    return sum(lst) / len(lst)


def crop_img(imgdata):
    # 5sheets
    # return imgdata[:, 240:255, 240:255]

    # 3sheets
    return imgdata[:, 235:260, 235:260]


def get_files(grating_width):
    # 3 sheets
    return {
        'confocal': f'/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/grating_images_3sheets_1nm/raw_imgs/grating_{grating_width}nm_out.tif',
        'sim': f'/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/grating_images_3sheets_1nm/processed_images/grating_{grating_width}nm_sim.tif',
        'rcan': f'/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/grating_images_3sheets_1nm/processed_images/grating_{grating_width}nm_out_final.tif',
    }

    # 5 sheets
    # return {
    #     'confocal': f'/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/grating_images_5sheets_2nm/raw_imgs/grating_5_sheet_{grating_width}nm_out.tif',
    #     'sim': f'/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/grating_images_5sheets_2nm/processed_images/grating_5_sheet_{grating_width}nm_sim.tif',
    #     'rcan': f'/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/grating_images_5sheets_2nm/processed_images/grating_5_sheet_{grating_width}nm_out_final.tif',
    # }


def get_peak_troths(impath):
    n = os.path.basename(impath)

    # 3 sheets
    grating_spacing = int(n.split('_')[1].replace('nm', ''))

    # 5 sheets
    # grating_spacing = int(n.split('_')[3].replace('nm', ''))

    peak_width = peak_widths[grating_spacing]

    imgdata = imread(impath)
    if imgdata.max() > 1:
        imgdata = imgdata / 255

    imgdata = crop_img(imgdata)

    imgdata[imgdata < 0] = 0
    series = np.mean(imgdata, axis=(1, 2))

    peaks, _ = signal.find_peaks(series, height=0.1, width=peak_width)
    peak_vals = [series[p] for p in peaks]

    data = dict()
    # data[n] = series
    # pd.DataFrame(data).plot.line()
    # plt.ylabel('Mean pixel value [0, 1]')
    # plt.xlabel('Z')
    # plt.title(n)
    # [plt.vlines(x=p, ymin=0, ymax=series[p]) for p in peaks]

    troth_vals = []
    for p, p2 in zip(peaks, peaks[1:]):
        subseries = series[p:p2 + 1]
        troth_val = subseries.min()
        troth_pos = subseries.argmin() + p
        plt.vlines(x=troth_pos, ymin=0, ymax=troth_val, colors='red')
        troth_vals.append(troth_val)
    # plt.show()
    # input()
    try:
        # 3 slice
        pairs = [
            (peak_vals[0], troth_vals[0]),
            (peak_vals[1], troth_vals[0]),
            (peak_vals[1], troth_vals[1]),
            (peak_vals[2], troth_vals[1]),
        ]

        # 5 slice
        # pairs = [
        #     (peak_vals[0], troth_vals[0]),
        #     (peak_vals[1], troth_vals[0]),
        #     (peak_vals[1], troth_vals[1]),
        #     (peak_vals[2], troth_vals[1]),
        #     (peak_vals[2], troth_vals[2]),
        #     (peak_vals[3], troth_vals[2]),
        #     (peak_vals[3], troth_vals[3]),
        #     (peak_vals[4], troth_vals[3]),
        # ]
        lmax = avg(peak_vals)
        lmin = avg(troth_vals)
        return (lmax-lmin) / (lmax+lmin)
    except IndexError:
        return 0

    vals = []
    for peak, troth in pairs:
        val = (peak - troth) / (peak + troth)
        vals.append(val)

    return sum(vals) / len(vals)


peak_widths = {
    600: None,
    700: None,
    800: 1,
    900: 1,
    1000: 1,
    1100: 1,
    1200: 2,
    1300: 3,
    1400: 3,
    1500: 3,
    1600: 3,
    1700: 3,
    1800: 3,
    1900: 3,
    2000: 3,
}

all_data = []

i = 0
for grating_spacing in peak_widths.keys():
    files = get_files(grating_spacing)
    data = {'grating_separation': grating_spacing}
    for k, f in files.items():
        data[k] = get_peak_troths(f)
        # input()
    all_data.append(data)
    i = i + 1

df = pd.DataFrame.from_records(all_data)
df = df.set_index('grating_separation')
print(df)
df.plot.line()
plt.grid(True)
plt.ylabel('Mean lmax-lmin / lmax+lmin')

plt.savefig('tmp.png')
plt.show()
df.to_csv('/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/results/gratings.csv')
