# Standard imports
import os
import torch
from data import *
import torch.utils.data as data
from utils.augmentations import VAEAugmentation
from torch.utils.tensorboard import SummaryWriter

# Third-party imports
from vae.model import VAE
from vae.utils import generate_psnr_table

# Constants
torch.manual_seed(0)


if __name__ == '__main__':
    # Create dirs if not already exist
    os.makedirs('vae/images', exist_ok = True)
    os.makedirs('vae/weights', exist_ok = True)
    os.makedirs('vae/tensorboard', exist_ok = True)
    os.makedirs('vae/psnrs', exist_ok = True)

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
    device          = torch.device('cuda') if use_cuda else torch.device('cpu')
    num_workers     = 2 if use_cuda else 4
    identifier      = f"ld{latent_dim}_bs{batch_size}_kl{kl_alpha}_lr{learning_rate}"

    # Initalize Tensorboard
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

    # Define Model
    vae = VAE(
        latent_dim = latent_dim,
        weight_decay = weight_decay,
        learning_rate = learning_rate,
        kl_alpha = kl_alpha,
        use_cuda = use_cuda,
        max_norm_gc = max_norm_gc,
    )

    # Load from checkpoint
    resume_from = 0
    # vae.load_weights(f"vae/weights/epoch_{resume_from}.path")

    # Train
    train_loss, val_loss = vae.train_valid(
        epochs = epochs,
        data_loader = data_loader,
        only_save_plots = True,
        tb_writer = tb_writer,
        resume_from = resume_from,
    )

    # Save PSNR variations table
    generate_psnr_table(
        vae = vae,
        dataset = data_loader.dataset,
        n = 100,
        times = 10,
        device = device,
        filename = f"vae/psnrs/{identifier}.csv"
    )
