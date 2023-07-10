import torch
from torch import nn
from typing import Optional


class Decoder(nn.Module):
    
    def __init__(self, latent_dim):
        super().__init__()

        self.model_1 = nn.Sequential(
            # Linear layer 1
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Linear layer 2
            nn.Linear(128, 64 * 16 * 16),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
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
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
        )


    def forward(self, x: torch.Tensor):
        x = self.model_1(x)
        x = x.reshape([-1, 64, 16, 16])
        x = self.model_2(x)
        x = torch.sigmoid(x)
        return x



class SkipDecoder(nn.Module):
    
    def __init__(self, latent_dim: int):
        super().__init__()

        self.model_1 = nn.Sequential(
            # Linear layer 1
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # Linear layer 2
            nn.Linear(128, 64 * 16 * 16),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
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
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
        )


    def forward(self, x: torch.Tensor, feature: torch.Tensor):
        x = self.model_1(x)                 # Apply linear and convolutional layers
        x = x.reshape([-1, 64, 16, 16])     # Unflatten
        x = torch.cat((x, feature), dim=1)  # Concat with the feature Tensor
        x = self.model_2(x)                 # Pass to second model
        x = torch.sigmoid(x)                # Scale outputs to be in [0, 1]
        return x
