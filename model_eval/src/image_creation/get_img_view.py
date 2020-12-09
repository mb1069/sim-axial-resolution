from PIL import Image
from tifffile import imread, imshow
import matplotlib.pyplot as plt
import os

imname = 'struct_10_out.tif'
model_name = 'final'
dirname = '/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/tmp_imgs'
outdir = dirname
images = {
    'out': os.path.join(dirname, 'raw_imgs', imname),
    'sim': os.path.join(dirname, 'processed_images', imname.replace('_out', '_sim')),
    'rcan': os.path.join(dirname, 'processed_images', imname.replace('_out', f'_out_{model_name}')),
}


def transform_img(impath, im_type):
    img = imread(impath)
    img = img / img.max()
    img = img[:, :, :]
    img = img[40:110, 240, 180:320].transpose()

    imshow(img)
    plt.title(im_type)
    plt.show()

    outname = os.path.join(outdir, f"{imname.replace('out.tif' ,'')}{im_type}_yz.png")
    Image.fromarray(img * 255).convert("L").save(outname)
    print(outname)



for im_type, impath in images.items():
    transform_img(impath, im_type)
