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

def parse_text_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    data = [l.strip() for l in lines]
    f.close()
    return data

class hpa_dataset(Dataset):
    def __init__(self, 
                 image_path, 
                 label_path,
                 image_base_path=None,
                 label_base_path=None,
                 image_size=224,
                 mode="train", 
                 transform=None,
                 one_hot=True,
                 sample_size=None,
                 train_split=0.7,
                 val_split=0.1
                 ):
        
        self.mode = mode
        self.transform = transform
        self.label_base_path = label_base_path
        self.image_base_path = image_base_path
        self.one_hot = one_hot
        random_sample_indices = None

        self.images = None
        self.labels = None
        self.labels_are_npy = False
        
        # Set default transforms: Resize.
        # NOTE: For labels, resize uses NEAREST to keep binary values.
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(
                    size=[image_size, image_size], 
                    interpolation=transforms.functional.InterpolationMode.BILINEAR
                ),
            ])
            
        # Load image paths from .txt file.
        if os.path.isfile(image_path) and image_path.endswith('.txt'):
            self.images = parse_text_file(image_path)
            if sample_size is not None and sample_size < len(self.images):
                random_sample_indices = np.random.choice(len(self.images), size=sample_size, replace=False)
                self.images = [self.images[i] for i in random_sample_indices]
            if image_base_path is not None:
                self.images = [os.path.join(image_base_path, path) for path in self.images]

            # Load corresponding label paths from .txt file, or load binary-valued labels directly from .npy.
            # Case .txt.
            print(label_path)
            if os.path.isfile(label_path) and label_path.endswith('.txt'):
                self.labels = parse_text_file(label_path)
                if label_base_path is not None:
                    self.labels = [os.path.join(label_base_path, path) for path in self.labels]
            # Case .npy.
            elif os.path.isfile(label_path) and label_path.endswith('.npy'):
                self.labels = torch.tensor(np.load(label_path))
                self.labels_are_npy = True
            
            
            # Apply the specific random indices, sampled above for self.images.
            if random_sample_indices is not None:
                self.labels = [self.labels[i] for i in random_sample_indices]

        else:
            raise ValueError(f'{input_path} must be an existing directory or txt file.')
        
        #print(len(self.images), len(self.labels))
        assert len(self.images) == len(self.labels)
        assert len(self.images) > 0
        
    def get_img_by_index(self, index):
        #img = read_image(self.images[index], ImageReadMode.RGB)
        img = transforms.ToTensor()(read_image(self.images[index], format='rgb'))
        return img
    
    def get_label_by_index(self, index):
        if self.labels_are_npy:
            label = self.labels[index]
        elif self.labels[index].endswith('.npz'):
            lbl = np.load(self.labels[index])
            label = transforms.ToTensor()(lbl['arr_0'].astype(np.int64))
        else:
            #label = read_image(self.labels[index], ImageReadMode.GRAY)
            label = transforms.ToTensor()(read_image(self.labels[index], format='binary'))

        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.get_img_by_index(index)
        label = self.get_label_by_index(index)

        # NOTE: transforms must be deterministic, to apply same to img and label.
        if self.transform:
            img = self.transform(img)
            img = (img - img.min())/(img.max() - img.min()).to(torch.float32)
        if self.transform:
            label = self.transform(label)
            label = (label - label.min())/(label.max() - label.min())
        
        if self.one_hot:
            label = F.one_hot(torch.squeeze(label).to(torch.int64))
            label = torch.moveaxis(label, -1, 0).to(torch.float32)

        short_id = os.path.splitext(os.path.basename(self.images[index]))[0]
        sample = {'image': img, 'label': label, 'id': short_id}
        return sample


