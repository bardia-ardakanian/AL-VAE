from data import *
from utility import *
from utils.augmentations import  VAEAugmentation
from layers.modules import MultiBoxLoss
from torch.utils.tensorboard import SummaryWriter
from ssd import build_ssd
import os
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# VAE
from vae.model import VAE


def train():
    voc_cfg = voc

    # Load VAE configuarions
    latent_dim      = vae_cfg['latent_dim']
    batch_size      = vae_cfg['batch_size']
    shuffle         = vae_cfg['shuffle']
    kl_alpha        = vae_cfg['kl_alpha']
    learning_rate   = vae_cfg['learning_rate']
    image_size      = vae_cfg['image_size']
    weight_decay    = vae_cfg['weight_decay']
    max_norm_gc     = vae_cfg['max_norm_gc']
    leaky_relu_ns   = vae_cfg['leaky_relu_ns']

    # Load VAE & SSD configurations
    use_cuda        = vae_ssd_cfg['use_cuda']
    iterations      = vae_ssd_cfg['iterations']
    ssd_loss_coef   = vae_ssd_cfg['ssd_loss_coef']
    use_tb          = vae_ssd_cfg['use_tb']
    num_workers     = 2 if use_cuda else 4
    identifier      = f"ssd_ld{latent_dim}_bs{batch_size}_kl{kl_alpha}_lr{learning_rate}"   # Must be unique

    # Initialize Tensorboard
    tb_writer = None
    if use_tb:
        tb_writer = SummaryWriter(f"vae/tensorboard/{identifier}")

    # Load data
    dataset = VOCDetection(
        root = 'data/VOCdevkit',
        transform = VAEAugmentation(image_size),
        sample_set = True,   # J1 images
        is_vae = True
    )
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        collate_fn = detection_collate,
        pin_memory = True,
    )

    # Define VAE
    vae = VAE(
        latent_dim = latent_dim,
        weight_decay = weight_decay,
        learning_rate = learning_rate,
        kl_alpha = kl_alpha,
        use_cuda = use_cuda,
        max_norm_gc = max_norm_gc,
        leaky_relu_ns = leaky_relu_ns,
    )
    vae.set_trainable() # Training mode

    # Define SSD
    ssd = build_ssd(
        phase = 'test',
        size = voc_cfg['min_dim'],
        num_classes = voc_cfg['num_classes']
    )
    criterion = MultiBoxLoss(
        num_classes = voc_cfg['num_classes'],
        overlap_thresh = 0.5,
        prior_for_matching = True,
        bkg_label = 0,
        neg_mining = True,
        neg_pos = 3,
        neg_overlap = 0.5,
        encode_target = False,
        use_gpu = use_cuda
    )
    if use_cuda:
        ssd = torch.nn.DataParallel(ssd)
        cudnn.benchmark = True
        ssd = ssd.cuda()
    ssd.eval()  # Set for evaluation

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(iterations):

        # load train data
        try:
            images, targets = next(batch_iterator)
        except:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if use_cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        # Validate SSD
        out = ssd(images)
        ssd_loss_l, ssd_loss_c = criterion(out, targets)
        ssd_loss = ssd_loss_l + ssd_loss_c  # Total loss

        # Train VAE
        _train_loss, _train_kl, _train_mse = vae.train_batch(
            x = images,
            extra_loss = (ssd_loss_coef * ssd_loss)
        )
        print(f"KL: {round(_train_kl.item(), 1)}", end = "\t")
        print(f"MSE: {round(_train_mse.item(), 1)}", end = "\t")
        print(f"Total: {round(_train_loss.item(), 1)}", end = "\n")

        # Validate VAE
        _valid_loss, _valid_kl, _valid_mse = vae.test_batch(
            x = images,
            extra_loss = (ssd_loss_coef * ssd_loss)
        )
        print(f"KL: {round(_valid_kl.item(), 1)}", end = "\t")
        print(f"MSE: {round(_valid_mse.item(), 1)}", end = "\t")
        print(f"Total: {round(_valid_loss.item(), 1)}", end = "\n")

        # Tensorbaord
        if use_tb:
            for name, metric in [
                    # KL
                    ('loss_kl/train', _train_kl),
                    ('loss_kl/valid', _valid_kl),
                    # MSE
                    ('loss_mse/train', _train_mse),
                    ('loss_mse/train', _valid_mse),
                    # Total
                    ('loss_total/train', _train_loss),
                    ('loss_total/train', _valid_loss),
                ]: tb_writer.add_scalar(name, metric.item(), iteration)

        # Checkpoint
        if iteration % 5000 == 0:
            vae.save_weights(f'vae/weights/{identifier}/iter_{repr(iteration)}.pth')


if __name__ == '__main__':
    # Create dirs if not already exist
    os.makedirs('vae/images', exist_ok = True)
    os.makedirs('vae/weights', exist_ok = True)
    os.makedirs('vae/tensorboard', exist_ok = True)
    os.makedirs('vae/psnrs', exist_ok = True)

    # training loop
    train()