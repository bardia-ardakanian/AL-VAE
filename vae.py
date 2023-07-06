import math
import torch
from typing import Tuple, List
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# Utilities
import torchvision
from torchvision.transforms import Compose, Resize
from torchsummary import summary
# pytorch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim

# Random seed for reproducible results
torch.manual_seed(0)

# Constants
IMG_WIDTH = 256     # Image Width
IMG_HEIGHT = 256    # Image Height
IMG_CHANNELS = 3    # Image Channels
IMG_COUNT = 5000    # Number of images to select from the dataset
TRAIN_DIR = 'data/mmsample/train2017'
VALID_DIR = 'data/mmsample/val2017'
KL_ALPHA = 10
LATENT_DIM = 128
EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Select device (CPU/GPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on '{DEVICE}'")



class CoCoDataset(Dataset):

    def __init__(self, files: List[str]) -> None:
        """ Reads a list of image paths and defines transformations """
        self.files = files
        self.transformations = Compose([
            Resize((IMG_WIDTH, IMG_HEIGHT), antialias=False)
        ])


    def __len__(self) -> int:
        """ Returns number of images """
        return len(self.files)


    def __getitem__(self, i: int):
        """ Reads and returns and image """
        img = torchvision.io.read_image(self.files[i])  # Load the image file
        img = self.transformations(img)                 # Apply transformations

        if img.shape[0] == 1:
            img = torch.cat([img] * 3)

        return img / 255.0


class VAELoss(nn.Module):

    def __init__(self, kl_alpha: float = 0.01):
        super(VAELoss, self).__init__()
        self.kl_alpha = kl_alpha


    def _kl(self, mean: float, logvar: float) -> float:
        """
            Compute the KL divergence between the prior and the approximate posterior

            Parameters:
                mean (float): mean of the approximate posterior
                logvar (float): log variance of the approximate posterior
            Returns:
                kl (float): KL divergence between the prior and the approximate posterior
        """
        return (logvar ** 2 + mean ** 2 - torch.log(logvar) - 1/2).sum()


    def _mse(self, x: torch.Tensor, x_hat: torch.Tensor) -> float:
        """
            Computes MSE between the actual and reconstructed images
            
            Parameters:
                x (torch.Tensor): Input tensor
                x_hat (torch.Tensor): Output tensor
            Returns:
                (float) Mean squared error
        """
        return F.mse_loss(x_hat, x, reduction = 'sum')


    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, mean: float, logvar: float) -> float:
        """
            Calculates the overal loss of the VAE
            
            Parameters:
                x (torch.Tensor): Input tensor
                x_hat (torch.Tensor): Output tensor
                mean (float): mean of the approximate posterior
                logvar (float): log variance of the approximate posterior
            Returns:
                 (float) Overal loss
        """
        mse = self._mse(x, x_hat)
        kl = self._kl(mean, logvar)
        loss = mse + kl
        return loss, kl, mse


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super(VariationalEncoder, self).__init__()

        self.model_1 = nn.Sequential(
            # Convolution layer 1
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Convolution layer 2
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Convolution layer 3
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Convolution layer 4
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

        self.model_2 = nn.Sequential(
            # Linear layer 1
            nn.Flatten(),
            nn.Linear(16384, 64 * 16 * 16),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

        # Mean and logvar layers
        self.mean_layer = nn.Linear(64 * 16 * 16, latent_dim)
        self.logvar_layer = nn.Linear(64 * 16 * 16, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            # Hack to get sampling on the GPU
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

        # Metrics
        self.mean = 0
        self.logvar = 0


    def forward(self, x, is_inference: bool = False):
        """
            Forward pass of the Encoder
            
            Parameters:
                x (torch.Tensor): input Tensor
                is_inference (bool): Uses Gaussian sampling in training mode and random sampling when in inference
        """
        
        x = x.to(DEVICE)
        feat = self.model_1(x)
        x = self.model_2(feat)

        # Calculate mean and standard deviation
        self.mean =  self.mean_layer(x)
        self.logvar = torch.exp(self.logvar_layer(x))

        if is_inference:
            # Random sampling
            return self.mean + self.logvar * torch.randn(self.mean.shape, device = DEVICE), feat
        else:
            # Gaussain sampling
            return self.mean + self.logvar * self.N.sample(self.mean.shape), feat


class Decoder(nn.Module):
    
    def __init__(self, latent_dim: int):
        super().__init__()

        self.model_1 = nn.Sequential(
            # Linear layer 1
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Linear layer 2
            nn.Linear(128, 64 * 16 * 16),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Unflatten(1, (64, 16, 16)),
        )

        self.model_2 = nn.Sequential(
            # Convolution layer 1
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Convolution layer 2
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Convolution layer 3
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Convolution layer 4
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1)
        )


    def forward(self, x, feat):
        x = self.model_1(x)     # Apply linear and convolutional layers
        x = torch.concat((x, feat), dim=1)
        x = self.model_2(x)
        x = torch.sigmoid(x)  # Scale outputs to be in [0, 1]
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x, is_inference: bool = False):
        x, feat = self.encoder(x, is_inference)
        x_hat = self.decoder(x, feat)
        return x_hat


def load_data():
    
    # Reads image files into a list
    train_files = [str(file) for file in Path(TRAIN_DIR).glob("*.jpg")]
    valid_files = [str(file) for file in Path(VALID_DIR).glob("*.jpg")]

    # Limit the dataset image counts
    train_files = train_files[:IMG_COUNT]
    valid_files = valid_files[:IMG_COUNT]

    # Use custom dataset loader
    train_dataset = CoCoDataset(train_files)
    valid_dataset = CoCoDataset(valid_files)

    # Define data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size = 48, 
        shuffle = True,
        drop_last = False, 
        num_workers = 2 if torch.cuda.is_available() else 4,
        pin_memory = True,  # avoid one implicit CPU-to-CPU copy
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size = 48, 
        shuffle = True, 
        drop_last = False, 
        num_workers = 2 if torch.cuda.is_available() else 4,
        pin_memory = True,  # avoid one implicit CPU-to-CPU copy
    )
    return train_dataset, train_loader, valid_dataset, valid_loader


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


def plot_reconstruction(model: VariationalAutoencoder, dataset: CoCoDataset, n: int = 5) -> None:
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
            rec_img = vae(img)

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


def train_epoch(model: VariationalAutoencoder, dataloader: DataLoader, optimizer: optim) -> float:
    """
        Trains the model for one epoch

        Parameters:
            model (VAE): The model to train
            dataloader (DataLoader): Dataloader to use for training
            optimizer (torch.optim): Optimizer to use for training
        Returns:
            train_loss (float): Average training loss for the epoch
    """
    print("training", end = "\t")

    # Set training mode for encoder and decoder
    model.train()
    loss = 0.0

    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for _, x in enumerate(dataloader):
        # load tensor to device
        x = x.to(DEVICE)
        # Run input through model
        x_hat = model(x)
        # Calculate batch loss (train_loss)
        _loss, kl, mse = vae_loss(x, x_hat, model.encoder.mean, model.encoder.logvar)
        # Backward pass
        optimizer.zero_grad()
        _loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        # Cumulate loss
        loss += _loss.item()

        print("=", end = "")

    print(">", end = "\t")
    return loss / len(dataloader.dataset)


def test_epoch(vae: VariationalAutoencoder, dataloader: DataLoader) -> float:
    """
        Test the model for one epoch

        Parameters:
            model (VAE): The model to test
            dataloader (DataLoader): Dataloader to use for testing
            verbose (bool): Whether to print the loss for each batch
        Returns:
            val_loss (float): Average validation loss for the epoch
    """
    print("validating", end = "\t")

    # Set evaluation mode for encoder and decoder
    vae.eval()
    loss = 0.0

    with torch.no_grad(): # No need to track the gradients
        for _, x in enumerate(dataloader):
            # load tensor to device
            x = x.to(DEVICE)
            # Run input through model
            x_hat = vae(x)
            # Calculate batch loss (test_loss)
            _loss, kl, mse = vae_loss(x, x_hat, vae.encoder.mean, vae.encoder.logvar)
            loss += _loss.item()

            print("=", end = "")

    print(">", end = "\t")
    return loss / len(dataloader.dataset)


def train(vae: VariationalAutoencoder, train_loader: DataLoader, valid_loader: DataLoader, epochs: int, optimizer: optim) -> Tuple[list, list]:
    """
        Train the model

        Parameters:
            vae (VariationalAutoencoder): VAE to train
            train_loader (DataLoader): Dataloader to use for training
            valid_loader (DataLoader): Dataloader to use for validation
            epochs (int): Number of epochs to train for
            optimizer (torch.optim): Optimizer to use for training
        Returns:
            None
    """

    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        # Train
        print(f"EPOCH {epoch + 1})", end = "\t")
        _train_loss = train_epoch(vae, train_loader, optimizer)
        print(f"loss: {round(_train_loss, 1)}")
        train_loss.append(_train_loss)

        # Test
        print(f"EPOCH {epoch + 1})", end = "\t")
        _val_loss = test_epoch(vae, valid_loader)
        print(f"loss: {round(_val_loss, 1)}")
        val_loss.append(_val_loss)

        plot_reconstruction(vae, valid_dataset, n = 10)
        print("")
    return train_loss, val_loss


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


def plot_random_reconstructions(model: VariationalAutoencoder, dataset: CoCoDataset, n: int = 3, times: int = 5) -> None:
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

    vae.encoder.eval()
    vae.decoder.eval()

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
                rec_img = vae(img + noise, is_inference = True)

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
    plt.show()



if __name__ == '__main__':
    train_dataset, train_loader, valid_dataset, valid_loader = load_data()

    # Define loss function
    vae_loss = VAELoss(KL_ALPHA)
    _ = vae_loss.to(DEVICE)

    # Define VAE
    vae = VariationalAutoencoder(LATENT_DIM)
    _ = vae.to(DEVICE)

    # Summary of the model structure
    # summary(vae, input_size = (IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT))

    # Define optimizaer
    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )

    # Train model
    train_loss, val_loss = train(vae, train_loader, valid_loader, EPOCHS, optimizer)

    # Plot metrics
    plot_metrics(train_loss, val_loss)

    # Plot reconstructrion
    plot_random_reconstructions(vae, valid_dataset)