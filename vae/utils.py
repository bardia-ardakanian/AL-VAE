# Standard imports
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def calculate_psnr(image1: np.array, image2: np.array):
    """
        Peak signal-to-noise ratio
        
        Parameters:
            image1 (np.array): First iamge
            image2 (np.array): Second iamge
        Returns:
            (float) the PSNR score
    """
    image1 = np.array(image1)
    image2 = np.array(image2)
    # Calculate the mean squared error (MSE)
    mse = np.mean((image1 - image2) ** 2)
    # Calculate the maximum possible pixel value
    max_pixel_value = np.max(image1)
    # Calculate the PSNR using the MSE and maximum pixel value
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr


def plot_reconstruction(vae, dataset: Dataset, n: int = 5, device: torch.device = torch.device("cpu"), filename: str = None) -> None:
    """
        Plot the original and reconstructed images

        Parameters:
            model (VAE): The VAE models
            dataset (Dataset): Dataset to use samples from
            n (int): Number of images to plot
            device (torch.device): Whether to use CPU or Cuda
            filename (str): If spesified, the plot will onyl be saved at the given path, displays it otherwise.
        Returns:
            None
    """
    vae.set_validation()
    plt.figure(figsize = (10, 3))
    plt.title("Reconstructed Images | No Inference")

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = dataset[i][0].unsqueeze(0).to(device)

        # Reconstruct the image using the encoder and decoder
        with torch.no_grad():
            rec_img, _, _, _ = vae.predict(img.float(), is_inference=False)

        # Plot original images
        plt.imshow(img.cpu().squeeze().permute(1, 2, 0).numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Plot reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(rec_img.cpu().squeeze().permute(1, 2, 0).numpy())
        psnr = calculate_psnr(rec_img, img)
        ax.set_title(f'{psnr:0.2f}')

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        # plt.tight_layout()
        plt.show()


def plot_random_reconstructions(vae, dataset: Dataset, n: int = 5, times: int = 5, device: torch.device = torch.device("cpu"), filename: str = None) -> None:
    """
        Plot the original and randomly reconstructed images

        Parameters:
            model (VAE): The VAE models
            dataset (Dataset): Dataset to use samples from
            n (int): Number of images to plot
            times (int): Number of times to feed an image to the network
            device (torch.device): Whether to use CPU or Cuda
            filename (str): If spesified, the plot will onyl be saved at the given path, displays it otherwise.
        Returns:
            None
    """
    vae.set_validation()
    plt.title("Random Reconstruction | Inference")

    for i in range(n):
        ax = plt.subplot(n, times + 1, (i * (times + 1) + 1))
        img = dataset[i][0].unsqueeze(0).to(device)

        # Plot the original image
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        plt.imshow(img.cpu().squeeze().permute(1, 2, 0).numpy())
        ax.set_title('Original')

        H_img = img.cpu().squeeze().permute(1, 2, 0).numpy()

        for j in range(times):

            # Reconstruct the image using the encoder and decoder
            with torch.no_grad():
                rec_img, _, _, _ = vae.predict(img.float(), is_inference=True)

            # Calculate PSNR
            E_img = rec_img.cpu().squeeze().permute(1, 2, 0).numpy()
            psnr = calculate_psnr(E_img, H_img)

            # Plot the reconstructed images
            ax = plt.subplot(n, times+1, (i * (times + 1)) + j + 2)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            plt.imshow(rec_img.cpu().squeeze().permute(1, 2, 0).numpy())
            ax.set_title(f'{psnr:0.2f}')

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        # plt.tight_layout()
        plt.show()


def generate_psnr_table(vae, dataset: Dataset, n: int = 5, times: int = 5, device: torch.device = torch.device("cpu"), filename: str = None) -> pd.DataFrame:
    """
        Generates the PSNR table for further analysis

        Parameters:
            model (VAE): The VAE models
            dataset (Dataset): Dataset to use samples from
            n (int): Number of images to plot
            times (int): Number of times to feed an image to the network
            device (torch.device): Whether to use CPU or Cuda
            filename (str): Filename for the results to be saved
        Returns:
            (pd.DataFrame): PSNR DataFrame
    """
    vae.set_validation()
    data = {i:[] for i in range(times)}

    for i in range(n):
        img = dataset[i][0].unsqueeze(0).to(device)

        H_img = img.cpu().squeeze().permute(1, 2, 0).numpy()

        for j in range(times):

            # Reconstruct the image using the encoder and decoder
            with torch.no_grad():
                rec_img, _, _, _ = vae.predict(img.float(), is_inference=True)

            # Calculate PSNR
            E_img = rec_img.cpu().squeeze().permute(1, 2, 0).numpy()
            psnr = calculate_psnr(E_img, H_img)
            data[j].append(round(psnr, 2))
    
    df = pd.DataFrame(data)
    df['max'] = df[df.columns].max(axis=1)
    df['min'] = df[df.columns].min(axis=1)
    df['variation'] = df['max'] - df['min']

    if filename:
        df.to_csv(filename, index=False)
    return df
