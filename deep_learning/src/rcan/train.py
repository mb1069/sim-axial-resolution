import torch
import logging
import sys
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import argparse
from deep_learning.src.data_manager.dataset import DataSetManager
from deep_learning.src.rcan.eval import eval_net
from deep_learning.src.rcan.model import RCAN
import pandas as pd
learning_rate = 10E-6
validation_perc = 0.1

epochs_per_validation_round = 5


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1.double() - img2.double()) ** 2)
        return 20 * torch.log10(1 / torch.sqrt(mse))


def train_net(net, dataset_dir, epochs, batch_size, learning_rate, device, validation_perc, checkpoint_dir,
              preload_data, run_name):
    config = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    if hasattr(net, 'n_resgroups'):
        config.update({
            'rcan_groups': net.n_resgroups,
            'rcan_blocks': net.n_resblocks,
            'rcan_feats': net.n_feats,
            'rcan_kernel_size': net.kernel_size,
        })
    else:
        config.update({
            'rcan_groups': net.module.n_resgroups,
            'rcan_blocks': net.module.n_resblocks,
            'rcan_feats': net.module.n_feats,
            'rcan_kernel_size': net.module.kernel_size,
        })

    dataset = DataSetManager(dataset_dir, preload=preload_data)
    print(f'{len(dataset)} total datapoints')
    n_val = int(len(dataset) * validation_perc)
    print(f'{n_val} validation datapoints')
    n_train = len(dataset) - n_val
    print(f'{n_train} training datapoints')

    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              drop_last=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_batches = len(train_loader)
    print(f'{num_batches} training batches')

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    optimiser = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min')
    criterion = nn.MSELoss()

    training_logs = []

    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # Add empty channel to raw image_creation
                raw_imgs = batch['raw']
                out_imgs = batch['processed']
                raw_imgs = raw_imgs.to(device=device, dtype=torch.float32)
                out_imgs = out_imgs.to(device=device, dtype=torch.float32)

                out_imgs_pred = net(raw_imgs)

                loss = criterion(out_imgs_pred, out_imgs)
                epoch_loss.append(float(loss.item()))

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimiser.step()
                pbar.update(raw_imgs.shape[0])
                global_step += 1

        if epoch % 5 == 0:
            learning_rate *= 0.99
            if save_cp:
                try:
                    os.mkdir(checkpoint_dir)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                chkp_file = os.path.join(checkpoint_dir + f'CP_epoch{epoch + 1}.pth')
                torch.save(net, chkp_file)
                logging.info(f'Checkpoint {epoch + 1} saved !')

        epoch_data = {'epoch': epoch,
                      'train_loss': np.mean(epoch_loss),
                      'val_loss': None}
        if (epoch % epochs_per_validation_round) == 0:
            tqdm.write('Eval time')
            val_score = eval_net(net, val_loader, device, criterion)
            tqdm.write(f'Val score: {val_score}')
            scheduler.step(val_score)

            logging.info('Validation cross entropy: {}'.format(val_score))

            epoch_data['val_loss'] = val_score

        training_logs.append(epoch_data)
        tqdm.write(str(epoch_data))
        global_step += 1

    pd.DataFrame(training_logs).to_csv('training_log.csv', index=False)

    # writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--checkpoint-dir', default=os.path.join(os.path.dirname(__file__), 'checkpoints'))
    parser.add_argument('-dd', '--data-dir', default='/Volumes/Samsung_T5/uni/sim_data_3frames')
    parser.add_argument('-s', '--save-checkpoints', action='store_true')
    parser.add_argument('-t', '--test-size', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--groups', type=int, default=12)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('-p', '--preload-data', action='store_true')
    parser.add_argument('-l', '--logfile', default=os.path.join(os.path.dirname(__file__), 'out.log'))
    parser.add_argument('-r', '--rname', default='myrun')
    parser.add_argument('--nframes', default=3, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    outpath = os.path.join(os.getcwd(), args.rname + '.pth')
    if os.path.exists(outpath):
        print('Output path already exists.')
        quit()
    #
    # logging.basicConfig(filename=args.logfile,
    #                     level=logging.DEBUG)
    save_cp = args.save_checkpoints

    n_frames = args.nframes

    n_input_frames = n_frames
    n_imgs_per_frame = 7

    n_output_frames = 1

    n_imgs_per_frame_output = 3

    input_channels = n_imgs_per_frame * n_input_frames
    output_channels = n_output_frames * n_imgs_per_frame_output

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RCAN(input_channels, output_channels, n_frames, args.groups, args.blocks, 64, args.kernel_size)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1 and not args.test_size:
            devices = list([i for i, _ in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(','))])
            print(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
            net = nn.DataParallel(net, device_ids=devices)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        net = net.cuda()

    if args.test_size:
        from torchsummary import summary

        print('Summary!')
        summary(net, input_size=(1, 7 * args.nframes, 256, 256))
        quit()

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(args.data_dir):
        raise EnvironmentError(f'Data directory is missing, should be {args.data_dir}')

    try:
        train_net(net=net,
                  dataset_dir=args.data_dir,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=learning_rate,
                  device=device,
                  validation_perc=validation_perc,
                  checkpoint_dir=os.path.abspath(checkpoint_dir),
                  preload_data=args.preload_data,
                  run_name=args.rname)
    except KeyboardInterrupt:
        interrupt_fname = 'INTERRUPTED.pth'
        torch.save(net, interrupt_fname)
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    outpath = os.path.join(os.getcwd(), args.rname + '.pth')
    torch.save(net, outpath)
    print(outpath)
