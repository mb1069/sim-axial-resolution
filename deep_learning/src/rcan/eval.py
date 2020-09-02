import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np

MAX_IMAGE_UPLOADS = 20


def eval_net(net, loader, device, wandb_step, criterion):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    sample_imgs = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:

            raw_imgs, processed_imgs = batch['raw'], batch['processed']
            raw_names = batch['raw_name']
            processed_names = batch['processed_name']
            raw_imgs = raw_imgs.to(device=device, dtype=torch.float32)
            processed_imgs = processed_imgs.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred_imgs = net(raw_imgs)

            tot += criterion(pred_imgs, processed_imgs).item()

            if torch.cuda.is_available():
                pred_imgs = pred_imgs.cpu()
                processed_imgs = processed_imgs.cpu()

            for i, (raw_name, processed_name, pred, real) in enumerate(
                    zip(raw_names, processed_names, pred_imgs, processed_imgs)):
                shape = pred.shape
                pred = pred.reshape(shape[0] * shape[1], shape[2]).numpy()
                real = real.reshape(shape[0] * shape[1], shape[2]).numpy()
                sample_imgs.append(wandb.Image(pred, caption=f'Pred {i} ({raw_name})'))
                sample_imgs.append(wandb.Image(real, caption=f'Real {i} ({processed_name})'))
                if i > MAX_IMAGE_UPLOADS:
                    break
            pbar.update()

    wandb.log({"examples": sample_imgs}, step=wandb_step)

    return tot / n_val
