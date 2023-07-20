# Standard imports
import math
import torch
import torchvision
import numpy as np
from torch import Tensor
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage


class VAEDataset(Dataset):

    def __init__(self, files: List[str], image_size: int) -> 'VAEDataset':
        """
            Reads a list of image paths and defines lazy transformations

            Parameters:
                files (List[str]): List of image pathes
                image_size (int): Width and height of images
        """
        self.files = files
        self.transformations = Compose([
            ToPILImage("RGB"),
            Resize((image_size, image_size)),
            ToTensor(),
        ])


    def __len__(self) -> int:
        """
            Returns number of images

            Returns:
                (int): Number of files on the dataset
        """
        return len(self.files)


    def __getitem__(self, i: int) -> Tensor:
        """
            Reads and returns an image given the index

            Parameters:
                i (int): The index for which to return the image
            Returns:
                (Tensor): Image
        """
        img = torchvision.io.read_image(self.files[i])
        img = self.transformations(img)     # Apply transformations
        if img.shape[0] == 1:
            img = torch.cat([img] * 3)
        return img



def load_data(
        train_dirs: List[str],
        test_dirs: List[str] = [],
        batch_size: int = 32,
        num_images: int = 10000,
        image_size: int = 256,
        num_workers: int = 2
        ) -> Tuple[Dataset, DataLoader, Dataset, DataLoader]:
    """
        Loads the data

        Parameters:
            train_dirs (List[str]): List of filenames for training images
            test_dirs (List[str]): List of filenames for testing images
            batch_size (int): Number of samples per batch to load
            num_images (int): Number of images to load select for the dataset
            num_workers (int): Number of workers to use
        Returns:
            (Tuple[Dataset, DataLoader, Dataset, DataLoader]): train_dataset, train_loader, valid_dataset, valid_loader
    """
    def _(dirs: List[str]):
        files = []
        [files.extend([str(file) for file in Path(dir).glob("*.jpg")]) for dir in dirs]
        dataset = VAEDataset(files[:num_images], image_size)
        loader = DataLoader(
            dataset,                                # The dataset to load from
            batch_size = batch_size,                # How many samples per batch to load
            shuffle = True,                         # Whether to shuffle the data
            drop_last = False,                      # Drop the last incomplete batch
            num_workers = num_workers,              # Number of processes to use for loading the data
            pin_memory = True,                      # avoid one implicit CPU-to-CPU copy
        )
        return dataset, loader

    train_dataset, train_loader = _(train_dirs)     # Load training data
    test_dataset, test_loader = _(test_dirs)        # Load testing data
    return train_dataset, train_loader, test_dataset, test_loader


def plot_metrics(train_losses: List[float], valid_losses: List[float], filename: str = None) -> None:
    """
        Plots the metrics
        
        Parameters:
            train_losses (List[float]): List of train losses for each epoch
            valid_losses (List[float]): List of validation losses for each epoch
            filename (str): If spesified, the plot will onyl be saved at the given path, displays it otherwise.
        Returns:
            None
    """
    plt.figure(figsize = (20, 5))
    plt.title("Training and Validation Loss")
    plt.plot([loss.item() for loss in train_losses], label = "training")
    plt.plot([loss.item() for loss in valid_losses], label = "validation")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.xticks(range(len(train_losses)))
    plt.legend()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


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


def plot_reconstruction(vae, dataset: VAEDataset, n: int = 5, device: torch.device = torch.device("cpu"), filename: str = None) -> None:
    """
        Plot the original and reconstructed images

        Parameters:
            model (VAE): The VAE models
            dataset (VAEDataset): Dataset to use samples from
            n (int): Number of images to plot
            device (torch.device): Whether to use CPU or Cuda
            filename (str): If spesified, the plot will onyl be saved at the given path, displays it otherwise.
        Returns:
            None
    """
    plt.figure(figsize = (10, 3))

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = dataset[i].unsqueeze(0).to(device)
        vae.model.eval()

        # Reconstruct the image using the encoder and decoder
        with torch.no_grad():
            rec_img, _, _, _ = vae.predict(img.float(), is_inference=False)

        # Plot original images
        plt.imshow(img.cpu().squeeze().permute(1, 2, 0).numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n // 2:
            ax.set_title('Original Images')

        # Plot reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(rec_img.cpu().squeeze().permute(1, 2, 0).numpy())
        if i == n // 2:
            ax.set_title(f'Reconstructed Images')

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        # plt.tight_layout()
        plt.show()


def plot_random_reconstructions(vae, dataset: VAEDataset, n: int = 5, times: int = 5, device: torch.device = torch.device("cpu"), filename: str = None) -> None:
    """
        Plot the original and randomly reconstructed images

        Parameters:
            model (VAE): The VAE models
            dataset (VAEDataset): Dataset to use samples from
            n (int): Number of images to plot
            times (int): Number of times to feed an image to the network
            device (torch.device): Whether to use CPU or Cuda
            filename (str): If spesified, the plot will onyl be saved at the given path, displays it otherwise.
        Returns:
            None
    """
    vae.model.eval()

    for i in range(n):
        ax = plt.subplot(n, times + 1, (i * (times + 1) + 1))
        img = dataset[i].unsqueeze(0).to(device)

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
                rec_img, _, _, _ = vae.predict(img.float() + noise, is_inference=True)

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
