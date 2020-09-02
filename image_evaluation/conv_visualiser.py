import torch
from tifffile import imread, imwrite
import os
import numpy as np
import shutil
from tqdm import tqdm

results_dir = './results'
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

example_input_file = '/Users/miguelboland/Projects/uni/project_3/src/model_runner/comparison/46_struct_in.tif'
data = imread(example_input_file)
data = data / data.max()
data = data[np.newaxis, np.newaxis, 84:84+7, :, :]
tail_data = torch.from_numpy(data).float()


model_file = '/Users/miguelboland/Projects/uni/project_3/src/model_runner/comparison/MSE_2D_RCAN_w_3D_Conv.pth'
model = torch.load(model_file, map_location='cpu')
if isinstance(model, torch.nn.DataParallel):
    model = model.module
model.eval()

os.chdir(results_dir)
os.mkdir('head')
os.chdir('head')
for i, l in tqdm(enumerate(model.head), total=len(model.head)):
    tail_data = l(tail_data)
    data = tail_data.detach().numpy()
    data = data.squeeze()
    for f, layer in enumerate(data):
        imwrite(os.path.join(f'head_{i}_{f}.png'), layer)

shp = tail_data.shape
tail_data = tail_data.view(shp[0], shp[1], shp[3], shp[4])
os.chdir(os.pardir)
os.mkdir('body')
os.chdir('body')
for i, l in tqdm(enumerate(model.body), total=len(model.body)):
    tail_data = l(tail_data)
    data = tail_data.detach().numpy().squeeze()
    for f, layer in enumerate(data):
        imwrite(os.path.join(f'body_{i}_{f}.png'), layer / layer.max())
        # tqdm.write(f'body_{i}_{f}.png\t{layer / layer.max()}')
