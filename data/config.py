# config.py
import os.path
from pathlib import Path

# gets home dir cross platform
ROOT = os.path.expanduser("~")
HOME = os.getcwd()

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

# VAE configurations
vae_cfg = {
    'use_cuda': True,           # Whether to use Cuda or run on CPU
    'latent_dim': 1024,         # Latent dimentions in which to save Mu and Sigma
    'kl_alpha': 50,             # Kullback-Leibler divergence coefficient
    'learning_rate': 1e-4,      # Optimizer learning rate
    'weight_decay': 1e-5,       # Optimizer weight decay
    'batch_size': 32,           # Batch size
    'shuffle': True,            # Whether to shuffle images
    'epochs': 1000,             # Number of epochs to train on
    'image_size': 300,          # Width and height of each image
    'num_images': 1000,         # Number of images to select from both training and testing
    'max_norm_gc': 5,           # Gradient Clipping max_norm
    'leaky_relu_ns': 0.01,      # LeakyReLU negative slope
    'use_tb': True,             # Whether to use Tensorboard to plot losses
}
