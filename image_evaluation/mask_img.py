import os
import shutil
from tifffile import imread, imwrite

output_dir = '/Users/miguelboland/Projects/uni/project_3/src/model_runner/benchmark/masked_raw_imgs/'
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

in_img_dir = '/Users/miguelboland/Projects/uni/project_3/src/model_runner/benchmark/raw_imgs/'

in_img = '/Users/miguelboland/Projects/uni/project_3/src/model_runner/benchmark/raw_imgs/27_struct_in.tif'
out_img = in_img.replace('_in', '_out')

in_img_data = imread(in_img)
out_img_data = imread(out_img)

start_img = 14
end_img = 17
in_slice = in_img_data[start_img * 7:end_img * 7]
out_slice = out_img_data[start_img * 3:end_img * 3]

for i in range(in_slice.shape[0]):
    in_img_masked = in_slice.copy()
    in_img_masked[i] = 0
    in_fname = os.path.join(output_dir, f'27_struct_{i}_mask_in.tif')
    imwrite(in_fname, in_img_masked, compress=6)
    out_fname = in_fname.replace('_in', '_out')
    imwrite(out_fname, out_slice, compress=6)