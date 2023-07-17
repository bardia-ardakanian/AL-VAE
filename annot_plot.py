from data import *
from utils.augmentations import SSDAugmentation, VAEAugmentation
import torch.utils.data as data
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    # dataset config
    dataset_root = VOC_ROOT
    cfg = voc

    # dataset
    dataset = VOCDetection(root=dataset_root, transform=VAEAugmentation(cfg['min_dim']), sample=True)
    # dataloader
    data_loader = data.DataLoader(dataset, 2, num_workers=2, shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(0, cfg['max_iter']):

        # load train data
        try:
            images, targets = next(batch_iterator)
        except:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        image, target = images[0], targets[0]

        plot_image_with_annotations(image, target)

        break
