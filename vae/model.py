# Standard imports
import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List
from torch.utils.tensorboard import SummaryWriter

# Third-party imports
from vae.utils import plot_random_reconstructions, plot_reconstruction, generate_psnr_table


class VAECore(nn.Module):

    def __init__(self,
                 latent_dim: int,
                 use_cuda: bool,
                 leaky_relu_ns: float,
                ) -> 'VAECore':
        """
            Variational Autoencoder Core model

            Parameters:
                latent_dim (int): Latent sapece dimentionality
                use_cuda (bool): Wether to use Conda for Normal sampling
        """
        super(VAECore, self).__init__()
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.latent_dim = latent_dim

        # Normal distribution sampling
        self.N = torch.distributions.Normal(0, 1)
        if use_cuda:
            # Hack to get sampling on the GPU
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

        # Encoder layers
        self.encoder_1 = nn.Sequential(
            # Convolution layer 1
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(leaky_relu_ns, False),
            # Convolution layer 2
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(leaky_relu_ns, False),
            # Convolution layer 3
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leaky_relu_ns, False),
            # Convolution layer 4
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(leaky_relu_ns, False),
        )
        self.encoder_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14400, 64 * 16 * 16),
            nn.LeakyReLU(leaky_relu_ns, False),
        )

        # Reparameterization layers
        self.mean_layer = nn.Linear(64 * 16 * 16, latent_dim)
        self.logvar_layer = nn.Linear(64 * 16 * 16, latent_dim)

        # Decoder layers
        self.decoder_1 = nn.Sequential(
            # Linear layer 1
            nn.Linear(latent_dim, latent_dim * 4),
            nn.LeakyReLU(leaky_relu_ns, True),
            # Linear layer 2
            nn.Linear(latent_dim * 4, 64 * 16 * 16),
            nn.LeakyReLU(leaky_relu_ns, True),
        )
        self.decoder_2 = nn.Sequential(
            # Convolution layer 1
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leaky_relu_ns, True),
            # Convolution layer 2
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(leaky_relu_ns, True),
            # Convolution layer 3
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(leaky_relu_ns, True),
            # Convolution layer 4
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=0),
        )


    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """
            Reparameterize for inference

            Parameters:
                mean (Tensor): Mean Tensor
                logvar (Tensor): Logvar Tensor
            Returns:
                (Tensor): Z tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device)
        return eps * std + mean


    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
            Forward pass for the encoder

            Parameters:
                x (torch.Tensor): Input
            Returns:
                (Tuple[Tensor, Tensor]): mean and logvar Tensors
        """
        feat = self.encoder_1(x)
        x = self.encoder_2(feat)

        # Calculate mean and standard deviation
        mean =  self.mean_layer(x)
        logvar = torch.exp(self.logvar_layer(x))

        return mean, logvar


    def decode(self, x: Tensor) -> Tensor:
        """
            Forward pass for the decoder

            Parameters:
                x (torch.Tensor): Input
            Returns:
                (torch.Tensor): Reconstructed image
        """
        x = self.decoder_1(x)
        x = x.reshape([-1, 64, 16, 16])
        x = self.decoder_2(x)
        x = torch.sigmoid(x)
        return x


    def forward(self, x: Tensor, is_inference: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Forward pass for the VAE

            Parameters:
                x (torch.Tensor): Input image
                is_inference (bool): Whether the VAE is used for inference or training.
                It will use random sampling on inference and normal sampling otherwise.
            Returns:
                (Tuple[Tensor, Tensor, Tensor]): Reconstructed image, mean and logvar
        """
        mean, logvar = self.encode(x)

        if is_inference:
            # Random sampling (aka. reparameterization)
            z = mean + logvar * torch.randn(mean.shape, device = self.device)
        else:
            # Gaussain sampling
            z = mean + logvar * self.N.sample(mean.shape)

        return self.decode(z), mean, logvar



class VAE(object):

    def __init__(self,
                 latent_dim: int = 256,
                 weight_decay: float = 1e-5,
                 learning_rate: float = 1e-3,
                 kl_alpha: float = 0.1,
                 use_cuda: bool = False,
                 max_norm_gc: int = 5,
                 leaky_relu_ns: float = 0.01,
                 ) -> 'VAE':
        """
            VAE Constructor

            Parameters:
                latent_dim (int): Latent sapece dimentionality
                weight_decay (float): Regularization weights decay
                learning_rate (float): Learning rate
                kl_alpha (float): Kullback-Leibler divergence coefficient
                use_cuda (bool): Wether to use Conda for Normal sampling
        """
        self.kl_alpha = kl_alpha
        self.max_norm_gc = max_norm_gc
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

        # VAE model
        self.model = VAECore(
            latent_dim=latent_dim,
            use_cuda=use_cuda,
            leaky_relu_ns=leaky_relu_ns,
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay
        )

    def get_loss(self, x: Tensor, x_hat: Tensor, mean: Tensor, logvar: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Calculates loss

            parameters:
                x (torch.Tensor): Input image
                x_hat (torch.Tensor): Reconstructed image
                mean (torch.Tensor): Mean representation
                logvar (torch.Tensor): Logvar representation
            Returns:
                (Tuple[Tensor, Tensor, Tensor]): Loss, KL, MSE
        """

        # MSE
        mse = F.mse_loss(x_hat, x, reduction = 'sum')
        # KL
        kl = (logvar ** 2 + mean ** 2 - torch.log(logvar) - 1/2).sum()
        kl *= self.kl_alpha     # Apply coefficient
        return (mse + kl), kl, mse


    def predict(self, x: Tensor, is_inference: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
            Generates a reconstructed iamge and returns the loss parameters

            Parameters:
                x (torch.Tensor): Input image
                in_inference (bool): Whether to infer the image or not
            Returns:
                (Tuple[Tensor, Tensor, Tensor, Tensor]): Reconstructed iamge, total loss, KL and MSE
        """
        x_hat, mean, logvar = self.model(x, is_inference)
        loss, kl, mse = self.get_loss(x, x_hat, mean, logvar)
        return x_hat, loss, kl, mse


    def train_batch(self, x: Tensor, extra_loss: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Trains the model on a given batch

            Parameters:
                x (torch.Tensor): Input 4D batch <3, batch_size, img_width, img_height>
                extra_loss (float): Outside loss from an other model. Leave to None to ignore it
            Returns:
                (Tuple[float, float, float]): Training losses
        """
        x_hat, loss, kl, mse = self.predict(x, is_inference=False)
        # Calculate loss
        if extra_loss:
            loss += extra_loss     # Add outside loss to total VAE loss
        # Backpropagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm_gc)
        self.optimizer.step()
        return loss, kl, mse


    def test_batch(self, x: Tensor, extra_loss: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Trains the model on a given batch

            Parameters:
                x (torch.Tensor): Input 4D batch <3, batch_size, img_width, img_height>
                extra_loss (float): Outside loss from an other model. Leave to None to ignore it
            Returns:
                (Tuple[float, float, float]): Testing losses
        """
        x_hat, loss, kl, mse = self.predict(x, is_inference=True)
        # Calcualte loss
        if extra_loss:
            loss += extra_loss     # Add outside loss to total VAE loss
        return loss, kl, mse


    def train_epoch(self, dataloader: DataLoader, extra_loss: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Trains and epoch on the given dataloader

            Parameters:
                dataloader (DataLoader): The dataloader to train on
                extra_loss (torch.Tensor): Extra loss if spesified
            Returns:
                (Tuple[float, float, float]): Training losses
        """
        print("Training", end = " ")
        self.set_trainable()
        loss = kl = mse = 0.0

        for _, x in enumerate(dataloader):
            # Load tensor to device
            x = x[0].to(self.device)
            # Train and calculate comulicative loss
            _loss, _kl, _mse = self.train_batch(x, extra_loss)
            loss += _loss
            kl += _kl
            mse += _mse

            print("=", end = "")

        print(">", end = " ")
        return loss / len(dataloader.dataset), \
                kl / len(dataloader.dataset), \
                mse / len(dataloader.dataset),


    def test_epoch(self, dataloader: DataLoader) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Test the model for one epoch

            Parameters:
                dataloader (DataLoader): Dataloader to use for testing
            Returns:
                Tuple[float, float, float]: Testing losses
        """
        print("Testing", end = " ")
        self.set_validation()
        loss = kl = mse = 0.0

        with torch.no_grad(): # No need to track the gradients
            for _, x in enumerate(dataloader):
                # load tensor to device
                x = x[0].to(self.device)
                _loss, _kl, _mse = self.test_batch(x)
                loss += _loss
                kl += _kl
                mse += _mse
                print("=", end = "")

        print(">", end = " ")
        return loss / len(dataloader.dataset), \
                kl / len(dataloader.dataset), \
                mse / len(dataloader.dataset),


    def train_valid(self,
                    epochs: int,
                    train_data_loader: Optional[DataLoader] = None,
                    valid_data_loader: Optional[DataLoader] = None,
                    only_save_plots: bool = True,
                    tb_writer: SummaryWriter = None,
                    resume_from: int = 0,
                    identifier: str = None,
                    ) -> None:
        """
            Trains and evaluates the model

            Parameters:
                epochs (int): Number of epochs to train on
                train_data_loader (DataLoader): The dataloader used for training
                valid_data_loader (DataLoader): The dataloader used for validation
                tb_writer (SummaryWriter): Tensorboard writer
            Returns:
                None
        """
        for epoch in range(resume_from, resume_from + epochs + 1):
            print(f"\nEPOCH {epoch})")
            # Train
            _train_loss, _train_kl, _train_mse = self.train_epoch(train_data_loader)
            print(f"KL: {round(_train_kl.item(), 1)}", end = "\t")
            print(f"MSE: {round(_train_mse.item(), 1)}", end = "\t")
            print(f"Total: {round(_train_loss.item(), 1)}", end = "\n")

            # Validate
            _valid_loss, _valid_kl, _valid_mse = self.test_epoch(valid_data_loader)
            print(f"KL: {round(_valid_kl.item(), 1)}", end = "\t")
            print(f"MSE: {round(_valid_mse.item(), 1)}", end = "\t")
            print(f"Total: {round(_valid_loss.item(), 1)}", end = "\n")

            if tb_writer:
                for name, metric in [
                    # KL
                    ('loss_kl/train', _train_kl),
                    ('loss_kl/valid', _valid_kl),
                    # MSE
                    ('loss_mse/train', _train_mse),
                    ('loss_mse/train', _valid_mse),
                    # Total
                    ('loss_total/train', _train_loss),
                    ('loss_total/train', _valid_loss),
                ]: tb_writer.add_scalar(name, metric.item(), epoch)

            if epoch % 100 == 0:
                # Training data Reconstruction
                plot_reconstruction(
                    vae = self,
                    dataset = train_data_loader.dataset,
                    device = self.device,
                    filename = f"vae/images/{identifier}/train_recons/{epoch}.jpg" if only_save_plots else None
                )
                # Validation data Reconstruction
                plot_reconstruction(
                    vae = self,
                    dataset = valid_data_loader.dataset,
                    device = self.device,
                    filename = f"vae/images/{identifier}/valid_recons/{epoch}.jpg" if only_save_plots else None
                )
                # Random Training data Reconstruction
                plot_random_reconstructions(
                    vae = self,
                    dataset = train_data_loader.dataset,
                    device = self.device,
                    filename = f"vae/images/{identifier}/random_train_recons/{epoch}.jpg" if only_save_plots else None
                )
                # Random Validation data Reconstruction
                plot_random_reconstructions(
                    vae = self,
                    dataset = valid_data_loader.dataset,
                    device = self.device,
                    filename = f"vae/images/{identifier}/random_valid_recons/{epoch}.jpg" if only_save_plots else None
                )
            if epoch % 200 == 0:
                # Save current weights
                self.save_weights(f'vae/weights/{identifier}/epoch_{epoch}.pth')


    def save_weights(self, filename: str) -> None:
        """
            Saves weights of the model

            Parameters:
                filename (str): The name and path for which to save the weights
            Returns:
                None
        """
        torch.save(
            obj = self.model.state_dict(),
            f = filename
        )


    def load_weights(self, filename: str) -> None:
        """
            Loads pretrained weights of the model

            Parameters:
                filename (str): The name and path for which to save the weights
            Returns:
                None
        """
        weights = torch.load(filename)
        self.model.load_state_dict(weights)


    def set_trainable(self):
        """ Sets the model state to trainable """
        self.model.train()


    def set_validation(self):
        """ Set the model state to validation """
        self.model.eval()
