from utility import *
from data.config import *
from data.voc0712 import VOC_ROOT
from utils.augmentations import VAEAugmentation

if __name__ == "__main__":
    # dataset config
    dataset_root = VOC_ROOT
    cfg = voc

    j3_loader, remaining_loader = mix_remaining_split(
        root = 'data/VOCdevkit',
        transform = VAEAugmentation(300),
        batch_size = 1,
        is_vae = True,
        num_workers = 2,
        shuffle = True)

    print(f'''J3 Dataset Size: {len(j3_loader)}\nRemaining Dataset Size: {len(remaining_loader)}''')

    # create batch iterator
    batch_iterator = iter(j3_loader)
    for iteration in range(0, cfg['max_iter']):

        # load train data
        try:
            images, targets = next(batch_iterator)
        except:
            batch_iterator = iter(j3_loader)
            images, targets = next(batch_iterator)

        image, target = images[0], targets[0]

        plot_image_with_annotations(image, target, cfg['min_dim'])

        break
