import glob
from tqdm import tqdm
from model_eval.src.evaluation.compare import run_model_wrapper, load_model
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory')

model_file = '/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/models/final.pth'
dir_name = '/Users/miguelboland/Projects/uni/phd/sim-axial-resolution/model_eval/new_analysis/tmp_imgs'



model = load_model(model_file)

results_dir = os.path.join(dir_name, 'processed_images')
raw_dir = os.path.join(dir_name, 'raw_imgs')

for inp_file in tqdm(glob.glob(os.path.join(raw_dir, '*_in.tif'))):
    run_model_wrapper(model, model_file, inp_file, results_dir)
