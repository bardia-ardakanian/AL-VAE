# AL-VAE

## Overview

This repository contains the implementation of an active learning framework for object detection, utilizing a Variational Autoencoder (VAE). The core idea is to use the VAE to identify the most informative samples from an unlabeled dataset to improve the training of a Single Shot MultiBox Detector (SSD) model. The model achieves faster convergence and better accuracy by focusing on the most informative data.

## Features

- **Variational Autoencoder (VAE)**: Integrates a VAE to estimate the informativeness of images in the dataset.
- **Active Learning**: Utilizes an active learning approach to select data that will have the most impact on the SSD model training.
- **Dual-Scoring System**: Employs a reconstruction loss from the VAE and a loss from the SSD on unlabeled data to rank and prioritize images.

## Goal

To optimize the training of the Single Shot MultiBox Detector (SSD) for object detection by implementing deep active learning strategies.

## Approach

- **Integration of VAEs**: To evaluate the information content of images, providing a measure of their potential utility for improving the SSD model.
- **Selective Training**: Images are selected based on their informativeness score, as determined by the VAE and SSD model losses.

## Image Selection Criterion

The model ranks images using a dual-scoring system that combines:

1. The reconstruction loss from the VAE, indicates how well the image can be generated from its latent representation.
2. The detection loss from the SSD when applied to the unlabeled data.

<img width="964" alt="image" src="https://github.com/bardia-ardakanian/AL-VAE/assets/58801017/568e2885-a344-4e85-bfe3-6b4e4c80cdf8">
