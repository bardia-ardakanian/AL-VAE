from typing import Union, Tuple
from data import *
from utils.augmentations import SSDAugmentation, VAEAugmentation
import torch.utils.data as data
import matplotlib.pyplot as plt


def exclude_sample_split(
        root: str,
        transform: Union[VAEAugmentation, SSDAugmentation],
        batch_size: int,
        is_vae: bool = False,
        num_workers: int = 2,
        shuffle: bool = True,
    ) -> Tuple[data.DataLoader, data.DataLoader]:

    def _(is_j1: bool):
        """ Wrapper """
        dataset = VOCDetection(
            root = root,
            transform = transform,
            sample_set = is_j1,         # J1 Sample
            exclude_set = not is_j1,    # J2 Sample
            is_vae = is_vae             # VAE compatible loader
        )
        loader = data.DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = shuffle,
            collate_fn = detection_collate,
            pin_memory = True,
        )
        return loader

    j1_loader = _(True)
    j2_loader = _(False)
    return j1_loader, j2_loader


def plot_image_with_annotations(image, targets, dim=300):
    # Get the first image and convert it to a NumPy array
    image = image.numpy().transpose((1, 2, 0))

    # Plot the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Loop over the targets in the first batch and plot the bounding boxes and captions
    for target in targets:
        # Get the class label and bounding box coordinates
        label_idx = int(target[-1])
        label = VOC_CLASSES[label_idx]

        x_min = target[0] * dim
        y_min = target[1] * dim
        x_max = target[2] * dim
        y_max = target[3] * dim

        # Plot the bounding box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, linewidth=2, color='red')
        ax.add_patch(rect)

        # Plot the caption
        ax.text(x_min, y_min - 5, label, fontsize=12, color='white', ha='left', va='top',
                bbox=dict(facecolor='blue', alpha=0.5, edgecolor='blue', boxstyle='round'))

    # Show the plot
    plt.show()
