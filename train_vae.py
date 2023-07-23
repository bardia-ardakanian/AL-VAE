# Standard imports
import os
import torch
from data import *
import torch.utils.data as data
from utils.augmentations import VAEAugmentation

# Third-party imports
from vae.utils import plot_metrics, plot_random_reconstructions, load_data, generate_psnr_table
from vae.model import VAE

# Constants
torch.manual_seed(0)
COCO_TRAIN_DIR: str = 'data/mmsample/train2017'
COCO_VALID_DIR: str = 'data/mmsample/val2017'
VOC_TRAIN_DIR: str = 'data/VOCdevkit/VOC2012/JPEGImages'
VOC_VALID_DIR: str = 'data/VOCdevkit/VOC2007/JPEGImages'


if __name__ == '__main__':
    os.makedirs('results/vae', exist_ok = True)

    # Load data
    dataset = VOCDetection(
        root = 'data/VOCdevkit',
        transform = VAEAugmentation(vae_cfg['image_size']),
        sample_set = True,   # J1 images
        is_vae = True
    )
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = vae_cfg['batch_size'],
        num_workers = 2 if vae_cfg['use_cuda'] else 4,
        shuffle = True,
        collate_fn = detection_collate,
        pin_memory = True,
    )

    # Define Model
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

    # Train
    train_loss, val_loss = vae.train_valid(
        epochs = vae_cfg['epochs'],
        data_loader = data_loader,
        checkpoints = False,
        only_save_plots = True,
    )

    # Generate PSNR table
    generate_psnr_table(
        model = vae,
        dataset = data_loader.dataset,
        n = 3,
        times = 5,
        device = torch.device('cuda') if vae_cfg['use_cuda'] else torch.device('cpu')
    )

    # Plot metrics
    plot_metrics(train_loss, val_loss, filename = 'results/vae/metrics.jpg')

    # Plot reconstructrion
    plot_random_reconstructions(
        model = vae,
        dataset = data_loader.dataset,
        n = 3,
        times = 5,
        device = torch.device('cuda') if vae_cfg['use_cuda'] else torch.device('cpu')
    )

    # Save final model
    vae.save_weights(f'weights/vae.pth')
