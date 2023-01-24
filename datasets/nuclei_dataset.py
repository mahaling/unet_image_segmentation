import os
from glob import glob
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch.utils.data import Dataset
#import albumentations as A
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



class nuclei_dataset(Dataset):
    
    def __init__(self, data_dir, mode="train", image_size=224, img_ext="png", one_hot=True, transform=None, train_split=0.7, val_split=0.1):
        self.data_dir = data_dir
        self.transform = transform
        self.one_hot = one_hot
        self.image_size = image_size

        self.images = glob(os.path.join(self.data_dir, "images/*.{}".format(img_ext)))
        self.masks = glob(os.path.join(self.data_dir, "masks/*.{}".format(img_ext)))
        #print(self.data_dir, len(self.images), len(self.masks))
        assert(len(self.images) == len(self.masks))

        self.images.sort()
        self.masks.sort()

        self.train_split = train_split
        self.val_split = val_split

        # split the dataset
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

'''

class nuclei_dataset(Dataset):

    def __init__(self, data_dir, mode='train', image_size=224, one_hot=True, transform=None, train_split=0.7, val_split=0.1):
        self.data_dir = data_dir
        self.transform = transform
        self.one_hot = one_hot
        self.image_size = image_size
        
        self.data_dirs = glob(os.path.join(self.data_dir, "*/"))

        self.train_split = train_split
        self.val_split = val_split

        # split the dataset
        n = len(self.data_dirs)
        n_train, n_val = int(self.train_split * n), int(self.val_split * n)

        if mode == "train":
            self.data_dirs = self.data_dirs[:n_train]
        elif mode == "val":
            self.data_dirs = self.data_dirs[n_train:n_train+n_val]
        elif mode == "test":
            self.data_dirs = self.data_dirs[n_train+n_val:]

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=[self.image_size, self.image_size], 
                    interpolation=transforms.functional.InterpolationMode.NEAREST),
            ])

    


    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        data_at_idx = self.data_dirs[idx]
        data_name = data_at_idx.split("/")[-2]

        xp = glob(os.path.join(data_at_idx, "images/*.png"))[0]
        yp = glob(os.path.join(data_at_idx, "masks/*.png"))

        image = read_image(xp, ImageReadMode.RGB)
        
        masks = torch.zeros(len(yp), image.shape[1], image.shape[2])

        for i, y in enumerate(yp):
            mask_ = read_image(y, ImageReadMode.GRAY)
            masks[i, :, :] = mask_
        
        masks = masks.to(torch.int64)
        mask = torch.amax(masks, dim=0)

        #if self.transform:
        #    im = np.transpose(image.numpy(), (1,2,0))
        #    
        #    msk = np.transpose(masks.numpy(), (1,2,0))
        #    
        #    augmented = self.transform(image=im, mask=msk)
        #    image = torch.from_numpy(augmented['image'])
        #    masks = torch.from_numpy(augmented['mask']).to(torch.float)

        #    image = np.transpose(image, (2, 0, 1))
        #    masks = np.transpose(masks, (2, 0, 1))
        
        #if self.one_hot:
        #    return {'image': image, 'mask': masks}
        #else:
        #    mask = torch.amax(masks, dim=0).to(torch.float)
        #    return {'image': image, 'mask': mask}

        if self.transform:
            image = self.transform(image)
            image = (image - image.min())/(image.max() - image.min()).to(torch.float32)

            mask = self.transform(mask)
            mask = (mask - mask.min())/(mask.max() - mask.min())

        if self.one_hot:
            mask = F.one_hot(torch.squeeze(mask).to(torch.int64))
            mask = torch.moveaxis(mask, -1, 0).to(torch.float32)
        
        data = {'image': image, 'mask': mask}
        return data
'''