from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import time
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
# VAE
from vae.model import VAE
from vae.utils import plot_reconstruction, plot_random_reconstructions, load_data

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--sample', default=False,
                    help='Train on 1000 image sample images')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--tensorboard', default=True, type=str2bool,
                    help='Use tensorboard for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available() and args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def train():
    voc_cfg = voc

    dataset = VOCDetection(
        root = args.dataset_root,
        transform = SSDAugmentation(
            size = voc_cfg['min_dim'],
            mean = MEANS
        ),
        sample = True if args.sample else False
    )

    if args.tensorboard:
        from datetime import datetime
        from torch.utils.tensorboard import SummaryWriter
        run_name = f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'
        writer = SummaryWriter(os.path.join('runs', 'SSD', 'tensorboard', run_name))

    ssd_net = build_ssd('train', voc_cfg['min_dim'], voc_cfg['num_classes'])
    net = ssd_net

    # VAE
    vae = VAE(
        latent_dim = vae_cfg['latent_dim'],
        weight_decay = vae_cfg['weight_decay'],
        learning_rate = vae_cfg['learning_rate'],
        kl_alpha = vae_cfg['kl_alpha'],
        use_cuda = vae_cfg['use_cuda'],
        has_skip = vae_cfg['has_skip'],
        use_xavier = vae_cfg['use_xavier'],
        max_norm_gc = vae_cfg['max_norm_gc'],
    )
    vae.set_trainable() # Training mode

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.cuda:
        net = net.cuda()

    criterion = MultiBoxLoss(
        num_classes = voc_cfg['num_classes'],
        overlap_thresh = 0.5,
        prior_for_matching = True,
        bkg_label = 0,
        neg_mining = True,
        neg_pos = 3,
        neg_overlap = 0.5,
        encode_target = False,
        use_gpu = args.cuda
    )

    # loss counters
    det_loss_l = 0
    det_loss_c = 0
    vae_loss_kl = 0
    vae_loss_mse = 0
    epoch = 0
    epoch_size = len(dataset) // args.batch_size

    # Detector DataLoader
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        shuffle = True,
        collate_fn = detection_collate,
        pin_memory = True
    )

    # VAE DataLoader
    vae_train_loader, vae_valid_loader = load_data(
        train_dirs = ["data\VOCdevkit\VOC2012\JPEGImages"],
        valid_dirs = ["data\VOCdevkit\VOC2007\JPEGImages"],
        batch_size = vae_cfg['batch_size'],
        num_images = vae_cfg['num_images'],
        image_size = vae_cfg['image_size'],
        num_workers = 2 if vae_cfg['use_cuda'] else 4
    )

    # create batch iterator
    batch_iterator = iter(data_loader)
    vae_batch_train_iterator = iter(vae_train_loader)
    vae_batch_valid_iterator = iter(vae_valid_loader)
    for iteration in range(voc_cfg['max_iter']):
        if iteration % epoch_size == 0:
            epoch += 1

        # load train data
        try:
            images, targets = next(batch_iterator)
            # VAE
            vae_train_images = next(vae_batch_train_iterator)
            vae_valid_images = next(vae_batch_valid_iterator)
        except:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
            # VAE
            vae_batch_train_iterator = iter(vae_train_loader)
            vae_batch_valid_iterator = iter(vae_valid_loader)
            vae_train_images = next(vae_batch_train_iterator)
            vae_valid_images = next(vae_batch_valid_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            # VAE
            vae_train_images = Variable(vae_train_images.cuda())
            vae_valid_images = Variable(vae_valid_images.cuda())
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
            # VAE
            vae_train_images = Variable(vae_train_images)
            vae_valid_images = Variable(vae_valid_images)

        # forward
        t0 = time.time()
        out = net(vae_train_images)
        _det_loss_l, _det_loss_c = criterion(out, targets)
        _det_loss = _det_loss_l + _det_loss_c
        _vae_loss, _vae_loss_kl, _vae_loss_mse = vae.train_batch(
            x = vae_train_images,
            extra_loss = (vae_cfg['detector_loss_coef'] * _det_loss)
        )
        vae_loss_kl += _vae_loss_kl.item()
        vae_loss_mse += _vae_loss_mse.item()
        det_loss_l += _det_loss_l.data
        det_loss_c += _det_loss_c.data
        t1 = time.time()

        # Log
        if iteration % 100 == 0:
            print(f"EPOCH {epoch}\tIter {repr(iteration)} - Took {round(t1-t0, 4)})")
            print(f"\t- DET Loss)\tLoc: {det_loss_l}\tConf:\t{det_loss_c}")
            print(f"\t- VAE Loss)\tKL: {round(_vae_loss_kl.item())}\tMSE:\t{round(_vae_loss_mse.item())}\tTotal:{round(_vae_loss.item())}")

        # Plot
        if iteration != 0 and iteration % 2000 == 0:
            # Validate
            plot_random_reconstructions(
                vae = vae,
                dataset = vae_valid_loader.dataset,
                n = 3,
                times = 5,
                device = torch.device('cuda') if vae_cfg['use_cuda'] else torch.device('cpu'),
                filename = f'results/vae/recon_{epoch}_random.jpg'
            )
            # Reconstruct
            plot_reconstruction(
                model = vae,
                dataset = vae_train_loader.dataset,
                n = 3,
                times = 5,
                device = torch.device('cuda') if vae_cfg['use_cuda'] else torch.device('cpu'),
                filename = f'results/vae/recon_{epoch}.jpg',
            )

        if args.tensorboard:
            writer.add_scalar('losses/det_loc_loss', det_loss_l.data, iteration)
            writer.add_scalar('losses/det_conf_loss', det_loss_c.data, iteration)
            writer.add_scalar('losses/vae_kl_loss', vae_loss_kl, iteration)
            writer.add_scalar('losses/vae_mse_loss', vae_loss_mse, iteration)

        # Checkpoint
        if iteration != 0 and iteration % 5000 == 0:
            vae.save_weights(f'weights/vae_{repr(iteration)}.pth')

    # Final state
    vae.save_weights(f'weights/vae.pth')


if __name__ == '__main__':
    os.makedirs('weights/vae', exist_ok = True)

    # training loop
    train()