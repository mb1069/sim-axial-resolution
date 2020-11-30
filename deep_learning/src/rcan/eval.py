import torch
from tqdm import tqdm
import numpy as np

MAX_IMAGE_UPLOADS = 20


def eval_net(net, loader, device, criterion):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    losses = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            raw_imgs, processed_imgs = batch['raw'], batch['processed']
            raw_imgs = raw_imgs.to(device=device, dtype=torch.float32)
            processed_imgs = processed_imgs.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred_imgs = net(raw_imgs)

            losses.append(float(criterion(pred_imgs, processed_imgs).item()))

            pbar.update()

    return np.mean(losses)
