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

