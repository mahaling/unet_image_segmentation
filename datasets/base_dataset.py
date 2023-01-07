import os
from glob import glob
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def read_image(path, format="rgb"):
    assert os.path.isfile(path)

    f = open(path, "rb")
    if format.lower() == "rgb":
        img = Image.open(f).convert("RGB")
    elif format.lower() == "gray":
        img = Image.open(f).convert("L")
    elif format.lower() == "binary":
        img = Image.open(f).convert("1")
    return img

def parse_text_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    data = [l.strip() for l in lines]
    f.close()
    return data

class segmentation_dataset(Dataset):

    def __init__(self, image_files, mask_files, mode="train", image_size=224, one_hot=True, transform=None, train_split=0.7, val_split=0.1):

        self.transform = transform
        self.one_hot = one_hot
        self.image_size = image_size

        if os.path.isfile(image_files) and image_files.endswith('.txt'):
            self.images = parse_text_file(image_files)
        else:
            self.images = image_files

        if os.path.isfile(mask_files) and mask_files.endswith(".txt"):
            self.masks = parse_text_file(mask_files)
        else:
            self.masks = mask_files


        assert(len(self.images) == len(self.masks))

        self.train_split = train_split
        self.val_split = val_split

        n = len(self.images)
        n_train, n_val = int(self.train_split * n), int(self.val_split * n)

        if mode == "train":
            self.images = self.images[:n_train]
            self.masks = self.masks[:n_train]
        elif mode == "val":
            self.images = self.images[n_train:n_train+n_val]
            self.masks = self.masks[n_train:n_train+n_val]
        elif mode == "test":
            self.images = self.images[n_train+n_val:]
            self.masks = self.masks[n_train+n_val:]

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=[self.image_size, self.image_size], 
                    interpolation=transforms.functional.InterpolationMode.NEAREST),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        id = os.path.basename(self.images[idx])[:-4]

        image = transforms.ToTensor()(read_image(self.images[idx], format="rgb"))
        mask = transforms.ToTensor()(read_image(self.masks[idx], format="binary"))

        if self.transform:
            image = self.transform(image)
            image = (image - image.min())/(image.max() - image.min()).to(torch.float32)

            mask = self.transform(mask)
            mask = (mask - mask.min())/(mask.max() - mask.min())

        if self.one_hot:
            mask = F.one_hot(torch.squeeze(mask).to(torch.int64))
            mask = torch.moveaxis(mask, -1, 0).to(torch.float32)
        
        data = {'image': image, 'mask': mask, 'id': id}
        return data