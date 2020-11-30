import glob
import os
import matplotlib.pyplot as plt
from natsort import natsorted
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

images = natsorted(glob.glob('/Volumes/Samsung_T5/uni/noise_sim_data/processed_images/struct_10_noise_*_out_final.tif'), reverse=True)

frame = 41

frames = []
for i in images:
    im_noise = int(i.split('_')[-3])
    if im_noise not in [2048, 16]:
        continue
    im_data = imread(i)[:, 200:450, 175:425]
    im_data = im_data.mean(axis=0)
    # im_data = im_data / im_data.max()
    frames.append(im_data)
    imshow(im_data)
    plt.show()

montage = np.hstack(frames)
montage = montage / montage.max()
imwrite('/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/results/rcan_struct_10_noise_montage.tif', montage, compress=6)
imshow(montage)
plt.show()
quit()



imnames = ['in', 'sim', 'out', 'rcan']


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
