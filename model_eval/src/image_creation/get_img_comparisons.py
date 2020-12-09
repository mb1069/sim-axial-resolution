import os
from tifffile import imread, imshow
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import numpy as np
import imageio

from scipy.ndimage import zoom


def normalise(img):
    img[img<0] = 0
    img = img.astype(np.float)
    img = img / img.max()
    return img


def make_film(images):
    images = [normalise(i[:, 125:275, 125:275]) for i in images]

    print([i.max() for i in images])
    print([i.shape for i in images])
    film_data1 = np.concatenate(images[0:2], axis=2)
    print(film_data1.shape)
    film_data2 = np.concatenate(images[2:], axis=2)
    print(film_data2.shape)

    film_data = np.concatenate([film_data1, film_data2], axis=1)
    print(film_data.shape)

    film_data[film_data < 0] = 0
    print(film_data.shape)
    outpath = os.path.join(outdir, 'movie.gif')
    imageio.mimsave(outpath, film_data)
    quit()
    print(outpath)
    for frame in range(15, film_data.shape[0]):
        still_frame = film_data[frame]
        imshow(still_frame)
        plt.show()
        outpath = os.path.join(outdir, f'movie_still_{frame}.png')
        Image.fromarray(still_frame * 255).convert("L").save(outpath)

    print(outpath)
    quit()



dirname = '/Volumes/Samsung_T5/uni/sim/external_sim_images'
outdir = './tmp'

try:
    if len(os.listdir(outdir)) > 0:
        ans = input('Are you sure you want to wipe the empty dir? (y|n)')
        if ans != 'y':
            quit()
except FileNotFoundError:
    pass
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir)

in_imname = 'mto_n1_hd_in.tif'

in_img = os.path.join(dirname, 'raw_imgs', in_imname)
out_img = in_img.replace('_in', '_out')

sim_img = in_img.replace('_in', '_sim').replace('raw_imgs', 'processed_images')

rcan_model = 'final'

rcan_img = os.path.join(dirname, 'processed_images',
                        in_imname.replace('_in', '_out_').replace('.tif', rcan_model) + '.tif')

print(in_img)
print(rcan_img)
print(sim_img)
print(out_img)



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
make_film([in_img_data] + output_imgs)
