import torch
from torch import nn

# Select device (CPU/GPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on '{DEVICE}'")


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            # Convolution layer 1
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Convolution layer 2
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Convolution layer 3
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Convolution layer 4
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Linear layer 1
            nn.Flatten(),
            nn.Linear(14400, 64 * 16 * 16),
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
        x = x.to(DEVICE)
        x = self.model(x)
        self.mean =  self.mean_layer(x)
        self.logvar = torch.exp(self.logvar_layer(x))

        if is_inference:
            # Random sampling
            return self.mean + self.logvar * torch.randn(self.mean.shape, device = DEVICE)
        else:
            # Gaussain sampling
            return self.mean + self.logvar * self.N.sample(self.mean.shape)




class SkipEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super(SkipEncoder, self).__init__()

        self.model_1 = nn.Sequential(
            # Convolution layer 1
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Convolution layer 2
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Convolution layer 3
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # Convolution layer 4
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
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
