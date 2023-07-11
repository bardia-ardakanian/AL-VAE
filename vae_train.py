from data import *
from utils.augmentations import VAEAugmentation
import os
import time
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
from visdom import Visdom

# VAE
from vae.model.networks import VAE
from vae.utils.utilities import plot_reconstruction, plot_random_reconstructions, plot_metrics

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--j1', default=True,
                    help='Train on J1 images')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--tensorboard', default=True, type=str2bool,
                    help='Use tensorboard for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.visdom == True:
        viz = Visdom()

    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(
            root = args.dataset_root,
            transform = VAEAugmentation(size = 256)
        )

    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(
            root = args.dataset_root,
            transform = VAEAugmentation(size = 256)
        )

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    if args.tensorboard:
        from datetime import datetime
        # from torch.utils.tensorboard import SummaryWriter
        run_name = f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'
        # writer = SummaryWriter(os.path.join('runs', 'VAE', 'tensorboard', run_name))

    vae = VAE(has_skip = True)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        vae.load_weights(args.resume)

    # if args.cuda:
    #     vae.model = vae.model.cuda()

    # loss counters
    loss = 0
    loss_kl = 0
    loss_mse = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training VAE on:', dataset.name)
    print('Dataset size:', len(dataset))
    print('Using the specified args:')
    print(args)

    if args.visdom:
        vis_title = 'VAE.PyTorch on ' + dataset.name
        vis_legend = ['KL Loss', 'MSE Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend, viz)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend, viz)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    from vae.utils.data_loader import load_data, VOCDataset
    train_dataset, data_loader, valid_dataset, valid_loader = load_data(
        train_dirs = ["data\VOCdevkit\VOC2012\JPEGImages", "data\VOCdevkit\VOC2007\JPEGImages"],
        valid_dirs = ["data\VOCdevkit\VOC2007\JPEGImages"],
        batch_size = 32,
        dataset = VOCDataset
    )
    dataset = train_dataset
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, iter_plot, epoch_plot,
                            'append', viz, epoch_size=epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        try:
            # images, targets = next(batch_iterator)
            images = next(batch_iterator)
        except:
            batch_iterator = iter(data_loader)
            # images, targets = next(batch_iterator)
            images = next(batch_iterator)

        if args.cuda:
            # images = Variable(images.cuda())
            images = Variable(images)
            # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            # images = Variable(images)
            images = Variable(images)
            # targets = [Variable(ann, volatile=True) for ann in targets]

        # Train on batch
        t0 = time.time()
        _loss, _loss_kl, _loss_mse = vae.train_batch(images)
        t1 = time.time()
        # Comulative loss
        loss += _loss
        loss_kl += _loss_kl
        loss_mse += _loss_mse

        if iteration != 0 and iteration % 50 == 0:
            plot_random_reconstructions(vae.model, dataset, save_only=True, filename=f"results/recon_random_{iteration}.jpg")
            plot_reconstruction(vae.model, dataset, save_only=True, filename=f"results/recon_{iteration}.jpg")

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (_loss), end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_kl, loss_mse,
                            iter_plot, epoch_plot, 'append', viz)

        # if args.tensorboard:
        #     writer.add_scalar('losses/kl_loss', loss_kl, iteration)
        #     writer.add_scalar('losses/mse_loss', loss_mse, iteration)
        #     writer.add_scalar('losses/total_loss', loss, iteration)

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(vae.model.state_dict(), 'weights/vae_' + args.dataset + '_' +
                       repr(iteration) + '.pth')

    torch.save(vae.model.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def create_vis_plot(_xlabel, _ylabel, _title, _legend, viz):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type, viz, epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    # training loop
    train()
