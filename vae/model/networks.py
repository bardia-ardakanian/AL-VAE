import torch
import os
import numpy as np
from typing import Tuple, Optional
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# Third-party
from vae.utils.utilities import plot_reconstruction
from vae.loss import VAELoss
from vae.model.decoders import Decoder, SkipDecoder
from vae.model.encoders import Encoder, SkipEncoder

# Select device (CPU/GPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on '{DEVICE}'")

# Constants
KL_ALPHA = 1
LATENT_DIM = 256
EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32


class VariationalAutoencoder(nn.Module):

    def __init__(self, latent_dim: int, has_skip: bool):
        self.has_skip = has_skip
        super(VariationalAutoencoder, self).__init__()
        if has_skip:
            self.encoder = SkipEncoder(latent_dim)
            self.decoder = SkipDecoder(latent_dim)
        else:
            self.encoder = Encoder(latent_dim)
            self.decoder = Decoder(latent_dim)


    def forward(self, x, is_inference: bool = False):
        if self.has_skip:
            x, feat = self.encoder(x, is_inference)
            x_hat = self.decoder(x, feat)
        else:
            x = self.encoder(x, is_inference)
            x_hat = self.decoder(x)
        return x_hat


    def load_weights(self, base_file: str):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights from state dict...')
            self.load_state_dict(
                torch.load(
                    f = base_file,
                    map_location=lambda storage,
                    loc: storage
                )
            )
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



class VAE(object):

    def __init__(self, has_skip: bool = True) -> None:
        """ Constructor """
        # Model
        self.model = VariationalAutoencoder(LATENT_DIM, has_skip)
        _ = self.model.to(DEVICE)
        # Loss
        self.loss_calculator = VAELoss(KL_ALPHA)
        _ = self.loss_calculator.to(DEVICE)
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = LEARNING_RATE,
            weight_decay = WEIGHT_DECAY
        )


    def train(self, train_loader: DataLoader, dataset: Optional[Dataset], valid_loader: Optional[DataLoader], epochs: int) -> Tuple[list, list]:
        train_loss = []
        val_loss = []

        for epoch in range(epochs):
            # Train
            print(f"EPOCH {epoch + 1})", end = "\t")
            _train_loss, _train_kl, _train_mse = self.train_epoch(train_loader)
            print(f"loss: {round(_train_loss, 1)}")
            train_loss.append(_train_loss)

            # Test
            if valid_loader:
                print(f"EPOCH {epoch + 1})", end = "\t")
                _val_loss, _val_kl, _val_mse = self.test_epoch(valid_loader)
                print(f"loss: {round(_val_loss, 1)}")
                val_loss.append(_val_loss)

            # Plot reconstruction
            if dataset:
                plot_reconstruction(self.model, dataset, n = 10, save_only = True)
            print("")
        return train_loss, val_loss


    def predict(self, x: np.ndarray) -> Tuple[torch.Tensor, float, float, float]:
        """
            Runs the model for input images and returns predictrions and loss

            Parameters:
                x (np.ndarray): 4D images
            Returns:
                Tuple[np.ndarray, float]: Predictions and loss
        """
        x_hat = self.model(x)
        _loss, kl, mse = self.loss_calculator(x, x_hat, self.model.encoder.mean, self.model.encoder.logvar)
        return x_hat, _loss, kl, mse


    def train_batch(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
            Trains the model on a batch and returns the batch loss

            Parameters:
                x (np.ndarray): 4D batch
            Returns:
                train_loss (float): Training loss for the batch
        """
        x_hat, _loss, kl, mse = self.predict(x)
        # Backward pass
        self.optimizer.zero_grad()
        _loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        return _loss.item(), kl.item(), mse.item()


    def train_epoch(self, dataloader: DataLoader) -> float:
        """
            Trains the model for one epoch

            Parameters:
                dataloader (DataLoader): Dataloader to use for training
                optimizer (torch.optim): Optimizer to use for training
            Returns:
                train_loss (float): Average training loss for the epoch
        """
        print("training", end = "\t")

        # Set training mode for encoder and decoder
        self.model.train()
        loss = 0.0

        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for _, x in enumerate(dataloader):
            # load tensor to device
            x = x.to(DEVICE)
            # Train and calculate loss
            _loss, kl, mse = self.train_batch(x)
            loss += _loss

            print("=", end = "")

        print(">", end = "\t")
        return loss / len(dataloader.dataset), kl / len(dataloader.dataset), mse / len(dataloader.dataset),


    def test_epoch(self, dataloader: DataLoader) -> float:
        """
            Test the model for one epoch

            Parameters:
                dataloader (DataLoader): Dataloader to use for testing
                verbose (bool): Whether to print the loss for each batch
            Returns:
                val_loss (float): Average validation loss for the epoch
        """
        print("validating", end = "\t")

        # Set evaluation mode for encoder and decoder
        self.model.eval()
        loss = 0.0

        with torch.no_grad(): # No need to track the gradients
            for _, x in enumerate(dataloader):
                # load tensor to device
                x = x.to(DEVICE)
                x_hat, _loss, kl, mse = self.predict(x)
                loss += _loss.item()

                print("=", end = "")

        print(">", end = "\t")
        return loss / len(dataloader.dataset), kl / len(dataloader.dataset), mse / len(dataloader.dataset),


    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)
