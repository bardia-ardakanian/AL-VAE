from utils.data import *

if __name__ == "__main__":
    # dataset config
    dataset_root = VOC_ROOT
    cfg = voc

    exclude_dataset, exclude_loader, sample_dataset, sample_loader = exclude_sample_split(dataset_root=VOC_ROOT, transform=VAEAugmentation(cfg['min_dim']))

    print(f'''Exclude Dataset Size: {len(exclude_dataset)}\nSample Dataset Size: {len(sample_dataset)}''')

    # create batch iterator
    batch_iterator = iter(sample_loader)
    for iteration in range(0, cfg['max_iter']):

        # load train data
        try:
            images, targets = next(batch_iterator)
        except:
            batch_iterator = iter(sample_loader)
            images, targets = next(batch_iterator)

        image, target = images[0], targets[0]

        plot_image_with_annotations(image, target, cfg['min_dim'])

        break
