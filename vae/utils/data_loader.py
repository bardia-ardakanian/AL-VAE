import torch
from typing import List
from pathlib import Path
import torchvision
from torchvision.transforms import Compose, Resize
from torch.utils.data import DataLoader, Dataset

# Constants
IMG_SIZE = 256      # Image Width
IMG_CHANNELS = 3    # Image Channels
IMG_COUNT = 50    # Number of images to select from the dataset


class CoCoDataset(Dataset):

    def __init__(self, files: List[str]) -> None:
        """ Reads a list of image paths and defines transformations """
        self.files = files
        self.transformations = Compose([
            # Resize((IMG_SIZE, IMG_SIZE), antialias=False)
            Resize((IMG_SIZE, IMG_SIZE))
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



class VOCDataset(Dataset):

    def __init__(self, files: List[str]) -> None:
        """ Reads a list of image paths and defines transformations """
        self.files = files
        self.transformations = Compose([
            # Resize((IMG_SIZE, IMG_SIZE), antialias=False)
            Resize((IMG_SIZE, IMG_SIZE))
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



def load_data(train_dir: str, valid_dir: str, batch_size: int, dataset: Dataset):

    train_files = [str(file) for file in Path(train_dir).glob("*.jpg")]
    print(f"Loaded {len(train_files)} training images.")
    valid_files = [str(file) for file in Path(valid_dir).glob("*.jpg")]
    print(f"Loaded {len(valid_files)} validation images.")

    # Limit the dataset image counts
    train_files = train_files[:IMG_COUNT]
    valid_files = valid_files[:IMG_COUNT]

    # Use custom dataset loader
    train_dataset = dataset(train_files)
    valid_dataset = dataset(valid_files)

    # Define data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size, 
        shuffle = True,
        drop_last = False, 
        num_workers = 2 if torch.cuda.is_available() else 4,
        pin_memory = True,  # avoid one implicit CPU-to-CPU copy
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = False, 
        num_workers = 2 if torch.cuda.is_available() else 4,
        pin_memory = True,  # avoid one implicit CPU-to-CPU copy
    )
    return train_dataset, train_loader, valid_dataset, valid_loader
