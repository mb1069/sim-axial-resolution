from model_eval.src.evaluation import compare
from tifffile import imread
import os
import glob

compare.show_charts = True

for img in glob.glob('/Volumes/Samsung_T5/uni/external_sim_images/processed_images/mto_n1_hd_out_final.tif'):
    img_data = imread(img)
    print(img)
    imname = os.path.basename(img)
    res = compare.benchmark_fwhm(img_data, imname, 'final')
    for axis in [k for k in res.keys()]:
        res[axis] *= compare.pixel_sizes[axis]
    img2 = img.replace('_sim.tif', '_sim_param_estimate.tif')
    img_data2 = imread(img2)
    imname2 = os.path.basename(img2)
    res2 = compare.benchmark_fwhm(img_data2, imname2, 'sim')
    for axis in [k for k in res2.keys()]:
        res2[axis] *= compare.pixel_sizes[axis]
    print(sorted(res.items(), key=lambda r: r[0]))
    print(sorted(res2.items(), key=lambda r: r[0]))


