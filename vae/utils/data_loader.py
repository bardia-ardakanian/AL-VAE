import torch
import cv2
from typing import List
from pathlib import Path
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torch.utils.data import DataLoader, Dataset
import xml.etree.ElementTree as ET

# Constants
IMG_SIZE = 256      # Image Width
IMG_CHANNELS = 3    # Image Channels
IMG_COUNT = 5000    # Number of images to select from the dataset


class CoCoDataset(Dataset):

    def __init__(self, files: List[str]) -> None:
        """ Reads a list of image paths and defines transformations """
        self.files = files
        self.transformations = Compose([
            ToPILImage("RGB"),
            Resize((IMG_SIZE, IMG_SIZE)),
            ToTensor(),
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

        return img



class VOCDataset(Dataset):

    def __init__(self, files: List[str]) -> None:
        """ Reads a list of image paths and defines transformations """
        self.files = files
        self.transformations = Compose([
            ToPILImage("RGB"),
            Resize((IMG_SIZE, IMG_SIZE)),
            ToTensor(),
        ])


    def __len__(self) -> int:
        """ Returns number of images """
        return len(self.files)


    def __getitem__(self, i: int):
        """ Reads and returns and image """
        img = cv2.imread(self.files[i])
        img = self.transformations(img)

        if img.shape[0] == 1:
            img = torch.cat([img] * 3)

        return img



def load_data(train_dirs: List[str], valid_dirs: List[str], batch_size: int, dataset: Dataset):

    # Load training data
    train_files = []
    for dir in train_dirs:
        train_files.extend([str(file) for file in Path(dir).glob("*.jpg")])
    print(f"Loaded {len(train_files)} training images.")

    # Load validation data
    valid_files = []
    for dir in valid_dirs:
        valid_files.extend([str(file) for file in Path(dir).glob("*.jpg")])
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
