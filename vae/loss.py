import torch
from torch import nn
import torch.nn.functional as F

# Select device (CPU/GPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on '{DEVICE}'")


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