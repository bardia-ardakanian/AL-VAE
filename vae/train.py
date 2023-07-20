# Standard imports
import os
import torch
from data import *

# Third-party imports
from vae.utils import plot_metrics, plot_random_reconstructions, load_data
from vae.model import VAE

# Constants
torch.manual_seed(0)
COCO_TRAIN_DIR: str = 'data/mmsample/train2017'
COCO_VALID_DIR: str = 'data/mmsample/val2017'
VOC_TRAIN_DIR: str = 'data/VOCdevkit/VOC2012/JPEGImages'
VOC_VALID_DIR: str = 'data/VOCdevkit/VOC2007/JPEGImages'



if __name__ == '__main__':
    os.makedirs('results/vae')

    # Load data
    train_dataset, train_loader, valid_dataset, valid_loader = load_data(
        train_dirs = [VOC_TRAIN_DIR],
        test_dirs = [VOC_VALID_DIR],
        batch_size = vae_cfg['batch_size'],
        num_images = vae_cfg['num_images'],
        image_size = vae_cfg['image_size'],
        num_workers = 2 if vae_cfg['use_cuda'] else 4
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
        train_loader = train_loader,
        valid_loader = valid_loader,
        checkpoints = False,
        only_save_plots = True
    )

    # Plot metrics
    plot_metrics(train_loss, val_loss, filename = 'results/vae/metrics.jpg')

    # Plot reconstructrion
    plot_random_reconstructions(
        vae,
        valid_dataset,
        n = 3,
        times = 5,
        device = torch.device('cuda') if vae_cfg['use_cuda'] else torch.device('cpu')
    )

    # Save final model
    torch.save(
        obj = vae.model.state_dict(),
        f = f'weights/vae.pth'
    )