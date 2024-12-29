# import glob
import os
import random
import numpy as np
import pandas as pd
import torch
from .common import read_las
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

def image_transform(img, image_transform):
    if image_transform == "random":
        transform = transforms.RandomApply(
            torch.nn.ModuleList(
                [
                    transforms.RandomCrop(size=(128, 128)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.RandomAffine(
                        degrees=(30, 70),
                        translate=(0.1, 0.3),
                        scale=(0.5, 0.75),
                    ),
                ]
            ),
            p=0.3,
        )
    elif image_transform == "compose":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(128, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomAffine(
                    degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                ),
            ]
        )
    else:
        transform = None

    if transform is None:
        aug_image = None
    else:
        aug_image = transform(img)

    return aug_image
