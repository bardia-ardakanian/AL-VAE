# Standard imports
import os
import torch
from data import *
import torch.utils.data as data
from utils.augmentations import VAEAugmentation
from torch.utils.tensorboard import SummaryWriter
from utility import exclude_sample_split, mix_remaining_split

# Third-party imports
from vae.model import VAE
from vae.utils import generate_psnr_table

# Constants
torch.manual_seed(0)


if __name__ == '__main__':

    # Load VAE configuarions
    use_cuda        = vae_cfg['use_cuda']
    latent_dim      = vae_cfg['latent_dim']
    batch_size      = vae_cfg['batch_size']
    shuffle         = vae_cfg['shuffle']
    kl_alpha        = vae_cfg['kl_alpha']
    learning_rate   = vae_cfg['learning_rate']
    epochs          = vae_cfg['epochs']
    use_tb          = vae_cfg['use_tb']
    image_size      = vae_cfg['image_size']
    weight_decay    = vae_cfg['weight_decay']
    max_norm_gc     = vae_cfg['max_norm_gc']
    leaky_relu_ns   = vae_cfg['leaky_relu_ns']
    device          = torch.device('cuda') if use_cuda else torch.device('cpu')
    num_workers     = 2 if use_cuda else 4
    identifier      = f"ld{latent_dim}_bs{batch_size}_kl{kl_alpha}_lr{learning_rate}"   # Must be unique

    # Create dirs if not already exist
    os.makedirs(f'vae/images/{identifier}', exist_ok = True)
    os.makedirs(f'vae/weights/{identifier}', exist_ok = True)
    os.makedirs(f'vae/tensorboard/{identifier}', exist_ok = True)
    os.makedirs(f'vae/psnrs/{identifier}', exist_ok = True)

    # Initalize Tensorboard
    tb_writer = None
    if use_tb:
        tb_writer = SummaryWriter(f"vae/tensorboard/{identifier}")

    # Load data
    j3_loader, j3_prime_loader = mix_remaining_split(
        root = 'data/VOCdevkit',
        transform = VAEAugmentation(image_size),
        batch_size = batch_size,
        is_vae = True,
        num_workers = num_workers,
        shuffle = shuffle
    )
    print(len(j3_loader.dataset), len(j3_prime_loader.dataset))

    # Define Model
    vae = VAE(
        latent_dim = latent_dim,
        weight_decay = weight_decay,
        learning_rate = learning_rate,
        kl_alpha = kl_alpha,
        use_cuda = use_cuda,
        max_norm_gc = max_norm_gc,
        leaky_relu_ns = leaky_relu_ns,
    )

    # Load from checkpoint
    resume_from = 0
    # vae.load_weights(f"vae/weights/epoch_{resume_from}.pth")

    # Train
    train_loss, val_loss = vae.train_valid(
        epochs = epochs,
        train_data_loader = j3_loader,
        valid_data_loader = j3_prime_loader,
        tb_writer = tb_writer,
        resume_from = resume_from,
        identifier = identifier,
    )

    # Save PSNR variations tables
    generate_psnr_table(
        vae = vae,
        dataset = j3_loader.dataset,
        device = device,
        filename = f"vae/psnrs/{identifier}/seen.csv"
    )
    generate_psnr_table(
        vae = vae,
        dataset = j3_loader.dataset,
        device = device,
        filename = f"vae/psnrs/{identifier}/unseen.csv"
    )
