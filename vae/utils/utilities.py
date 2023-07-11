import math
import torch
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt

# Third-party imports
from vae.utils.data_loader import VOCDataset, CoCoDataset

# Select device (CPU/GPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on '{DEVICE}'")


def plot_metrics(train_loss: List[float], val_loss: List[float]) -> None:
    """
        Plots the metrics
        
        Parameters:
            train_loss (List[float]): List of train losses for each epoch
            test_loss (List[float]): List of test losses for each epoch
        Returns:
            None
    """
    plt.figure(figsize = (20, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_loss, label = "training")
    plt.plot(val_loss, label = "validation")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.xticks(range(len(train_loss)))
    plt.legend()
    plt.show()


def plot_reconstruction(model, dataset: CoCoDataset, n: int = 5, save_only: bool = False, filename: str = None) -> None:
    """
        Plot the original and reconstructed images

        Parameters:
            model (VAE): The VAE models
            dataset (torch.utils.data.Dataset): Dataset to use samples from
            n (int): Number of images to plot
        Returns:
            None
    """

    plt.figure(figsize = (10, 3))

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = dataset[i].unsqueeze(0).to(DEVICE)
        model.encoder.eval()
        model.decoder.eval()

        # Reconstruct the image using the encoder and decoder
        with torch.no_grad():
            rec_img = model(img.float())

        # Plot original images
        plt.imshow(img.cpu().squeeze().permute(1, 2, 0).numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n // 2:
            ax.set_title('Original images')

        # Plot reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().permute(1, 2, 0).numpy())  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n // 2:
            ax.set_title('Reconstructed images')

    plt.tight_layout()
    if save_only:
        plt.savefig('results/image.png' if not filename else filename, bbox_inches='tight')
    else:
        plt.show()


def calculate_psnr(img1: np.array, img2: np.array, border: int = 0):
    """
        Peak signal-to-noise ratio
        
        Parameters:
            img1 (np.array): First iamge
            img2 (np.array): Second iamge
            border (int): Border width of the images
        Returns:
            (float) the PSNR score
    """

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    # Remove border
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    # Calculate MSE
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)

    # Calculate PSNR
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 25.0
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out


def plot_random_reconstructions(model, dataset: CoCoDataset, n: int = 3, times: int = 5, save_only: bool = False, filename: str = None) -> None:
    """
        Plot the original and randomly reconstructed images

        Parameters:
            model (VAE): The VAE models
            dataset (torch.utils.data.Dataset): Dataset to use samples from
            n (int): Number of images to plot
            times (int): Number of times to feed an image to the network
        Returns:
            None
    """

    model.encoder.eval()
    model.decoder.eval()

    for i in range(n):
        ax = plt.subplot(n, times + 1, (i * (times + 1) + 1))
        img = dataset[i].unsqueeze(0).to(DEVICE)

        # Plot the original image
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        plt.imshow(img.cpu().squeeze().permute(1, 2, 0).numpy())
        ax.set_title('Original')

        H_img = img.cpu().squeeze().permute(1, 2, 0).numpy()

        for j in range(times):

            # Reconstruct the image using the encoder and decoder
            with torch.no_grad():
                noise = torch.randn_like(img) * 0 # Add noise
                rec_img = model(img.float() + noise, is_inference = True)

            # Calculate PSNR
            E_img = rec_img.cpu().squeeze().permute(1, 2, 0).numpy()
            psnr = calculate_psnr(E_img, H_img, border=0)

            # Plot the reconstructed images
            ax = plt.subplot(n, times+1, (i * (times + 1)) + j + 2)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            plt.imshow(rec_img.cpu().squeeze().permute(1, 2, 0).numpy())
            ax.set_title(f'{psnr:0.2f}')

    plt.tight_layout()
    if save_only:
        plt.savefig('results/image.png' if not filename else filename, bbox_inches='tight')
    else:
        plt.show()