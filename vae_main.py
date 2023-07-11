import torch

# Third-party imports
from vae.utils.utilities import plot_metrics, plot_random_reconstructions
from vae.utils.data_loader import load_data, VOCDataset, CoCoDataset
from vae.model.networks import VAE

# Random seed for reproducible results
torch.manual_seed(0)

# Constants
COCO_TRAIN_DIR = 'data/mmsample/train2017'
COCO_VALID_DIR = 'data/mmsample/val2017'
VOC_TRAIN_DIR = 'data/VOCdevkit/VOC2012/JPEGImages'
VOC_VALID_DIR = 'data/VOCdevkit/VOC2007/JPEGImages'
EPOCHS = 12
BATCH_SIZE = 32


if __name__ == '__main__':

    # Load COCO data
    # train_dataset, train_loader, valid_dataset, valid_loader = load_data(train_dir = COCO_TRAIN_DIR, valid_dir = COCO_VALID_DIR, batch_size = BATCH_SIZE, dataset = CoCoDataset)

    # Load VOC data
    train_dataset, train_loader, valid_dataset, valid_loader = load_data(
        train_dir = VOC_TRAIN_DIR,
        valid_dir = VOC_VALID_DIR,
        batch_size = BATCH_SIZE,
        dataset = VOCDataset
    )

    # VOC
    vae = VAE(False)
    train_loss, val_loss = vae.train(train_loader, train_dataset, valid_loader, EPOCHS)

    # Plot metrics
    plot_metrics(train_loss, val_loss)

    # # Plot reconstructrion
    plot_random_reconstructions(vae, valid_dataset)