import os
import matplotlib.pyplot as plt
from tifffile import imread, imshow, imwrite
from scipy.ndimage import zoom
import numpy as np


def normalise(img):
    img[img < 0] = 0
    img = img.astype(np.float)
    img = img / img.max()
    return img


def tile_images(images):
    film_data1 = np.concatenate(images[0:2], axis=1)
    film_data2 = np.concatenate(images[2:], axis=1)
    film_data = np.concatenate([film_data1, film_data2], axis=0)
    return film_data


# dirname = '/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/grating_images_3sheets_1nm/'
# in_imname = 'grating_1500nm_in.tif'

dirname = '/Volumes/Samsung_T5/uni/external_sim_images/'
in_imname = 'mto_n1_hd_in.tif'

in_img = os.path.join(dirname, 'raw_imgs', in_imname)
out_img = in_img.replace('_in', '_out')

sim_img = in_img.replace('_in', '_sim').replace('raw_imgs', 'processed_images')

rcan_model = 'final'

rcan_img = os.path.join(dirname, 'processed_images',
                        in_imname.replace('_in.tif', f'_out_{rcan_model}.tif'))
tmp_dirname = '/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/src/tmp'
out_img_zoom = os.path.join(tmp_dirname, os.path.basename(out_img.replace('.tif', '_zoom.tif')))
rcan_img_zoom = os.path.join(tmp_dirname, os.path.basename(rcan_img.replace('.tif', '_zoom.tif')))
sim_img_zoom = os.path.join(tmp_dirname, os.path.basename(sim_img.replace('.tif', '_zoom.tif')))
in_img_zoom = os.path.join(tmp_dirname, os.path.basename(in_img.replace('.tif', '_zoom.tif')))

if not all([os.path.exists(p) for p in [out_img_zoom, rcan_img_zoom, sim_img_zoom, in_img_zoom]]):

    out_img_data = imread(out_img)
    rcan_img_data = imread(rcan_img)
    sim_img_data = imread(sim_img)
    in_img_data = imread(in_img)

    in_img_data = zoom(in_img_data, (1, 2, 2))
    in_img_shape = in_img_data.shape
    out_img_shape = out_img_data.shape
    ratio = [i / o for o, i in zip(out_img_shape, in_img_shape)]
    output_imgs = [sim_img_data, out_img_data, rcan_img_data]

    output_imgs = [zoom(i, ratio) for i in output_imgs]
    imwrite(sim_img_zoom, output_imgs[0], compress=6)
    imwrite(out_img_zoom, output_imgs[1], compress=6)
    imwrite(rcan_img_zoom, output_imgs[2], compress=6)
    imwrite(in_img_zoom, in_img_data, compress=6)
else:
    in_img_data = imread(in_img_zoom)
    output_imgs = [imread(i) for i in (sim_img_zoom, out_img_zoom, rcan_img_zoom)]

frame = 35

imnames = ['in', 'sim', 'out', 'rcan']
images = [in_img_data, *output_imgs]


def process_challenge_data(imname, image):
    image = normalise(image)
    image = np.fliplr(np.flipud(image))

    # lateral crop
    image = image[:, 250:400, 125:275]
    # sum along y
    image = image.mean(axis=1)
    image = image.astype(np.float32)
    imshow(image)
    plt.show()
    return image

# def process(imname, image):
#     # image = np.fliplr(np.flipud(image))
#
#     # lateral crop
#     image = image[80:190, 220:270, 230:260]
#     image = normalise(image)
#
#     # sum along y
#     image = image.mean(axis=2)
#     return image

min_z = min([i.shape[0] for i in images])
images = [i[:min_z, :, :] for i in images]

images = [process_challenge_data(imname, i) for imname, i in zip(imnames, images)]


tiled_image = tile_images(images)
imshow(tiled_image)
# plt.savefig('/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/results/gratings.png')
# imwrite('/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/results/challenge.tif', tiled_image.astype(np.float32), compress=6)
plt.show()
