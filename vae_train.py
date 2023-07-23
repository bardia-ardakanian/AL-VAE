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
from vae.utils import plot_reconstruction, plot_metrics, plot_random_reconstructions

DETECTOR_LOSS_COEF = 1


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--j1', default=True,
                    help='Train on J1 images')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():

    # Load detector datasets
    if args.dataset == 'COCO':
        cfg = coco
        dataset = COCODetection(
            root = args.dataset_root,
            transform = SSDAugmentation(
                size = cfg['min_dim'],
                mean = MEANS
            )
        )
    elif args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(
            root = args.dataset_root,
            transform = SSDAugmentation(
                size = cfg['min_dim'],
                means = MEANS
            ),
            j1 = True
        )

    net = build_ssd(
        phase = 'train',
        size = cfg['min_dim'],
        num_classes = cfg['num_classes']
    )
    vae = VAE(
        weight_decay = vae_cfg['weight_decay'],
        kl_alpha = vae_cfg['kl_alpha'],
        learning_rate = vae_cfg['learning_rate'],
        latent_dim = vae_cfg['latent_dim'],
        has_skip = vae_cfg['has_skip']
    )

    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        # Load models on GPU
        net = net.cuda()
        vae.model = vae.model.cuda()

    criterion = MultiBoxLoss(
        num_classes = cfg['num_classes'],
        overlap_thresh = 0.5,
        prior_for_matching = True,
        bkg_label = 0,
        neg_mining = True,
        neg_pos = 3,
        neg_overlap = 0.5,
        encode_target = False,
        use_gpu = args.cuda
    )

    epoch = 0
    epoch_size = len(dataset) // args.batch_size

    # Load Detector data
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        shuffle = True,
        collate_fn = detection_collate,
        pin_memory = True
    )
    # Load VAE data
    vae_train_dataset, vae_train_loader, vae_valid_dataset, vae_valid_loader = load_data(
        train_dirs = ["data\VOCdevkit\VOC2012\JPEGImages", "data\VOCdevkit\VOC2007\JPEGImages"],
        valid_dirs = ["data\VOCdevkit\VOC2007\JPEGImages"],
        batch_size = vae_cfg['batch_size'],
        dataset = VOCDataset
    )

    # create batch iterator
    batch_iterator = iter(data_loader)
    vae_batch_train_iterator = iter(vae_train_loader)
    vae_batch_valid_iterator = iter(vae_valid_loader)
    for iteration in range(0, vae_cfg['max_iter']):
        if iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1

        # Load image batch
        try:
            images, targets = next(batch_iterator)
            vae_images = next(vae_batch_train_iterator)
        except:
            batch_iterator = iter(data_loader)
            vae_batch_train_iterator = iter(vae_train_loader)
            images, targets = next(batch_iterator)
            vae_images = next(vae_batch_train_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            vae_images = Variable(vae_images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            vae_images = Variable(vae_images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        # forward
        t0 = time.time()
        out = net(vae_images)                       # Get Detector outputs
        loss_l, loss_c = criterion(out, targets)    # Calculate localization and confidence loss
        det_loss = loss_l + loss_c                  # Calculate total loss
        vae_loss, vae_loss_kl, vae_loss_mse = vae.train_batch(
            x = vae_images,
            extra_loss = (DETECTOR_LOSS_COEF * det_loss)
        )
        t1 = time.time()

        if iteration != 0 and iteration % 2000 == 0:
            # Validate
            plot_random_reconstructions(
                model = vae.model,
                dataset = vae_valid_dataset,
                save_only = True,
                filename = f"results/recon_{iteration}_random.jpg"
            )
            # Reconstruct
            plot_reconstruction(
                model = vae.model,
                dataset = vae_train_dataset,
                save_only = True,
                filename = f"results/recon_{iteration}.jpg"
            )

        # Log
        if iteration % 50 == 0:
            print(f"EPOCH {epoch}\tIter {repr(iteration)} - Took {round(t1-t0, 4)})")
            print(f"\t- Detector Loss)\tLoc: {loss_l}\tConf:\t{loss_c}\tTotal:{det_loss}")
            print(f"\t- VAE Loss)\tKL: {vae_loss_kl}\tMSE:\t{vae_loss_mse}\tTotal:{vae_loss}")

        # Checkpoint
        if iteration != 0 and iteration % 5000 == 0:
            print("\t- Saving VAE weights...")
            torch.save(
                obj = vae.model.state_dict(),
                f = 'weights/vae_' + args.dataset + '_' + repr(iteration) + '.pth'
            )

    torch.save(
        obj = vae.model.state_dict(),
        f = args.save_folder + '' + args.dataset + '.pth'
    )



if __name__ == '__main__':
    train()
