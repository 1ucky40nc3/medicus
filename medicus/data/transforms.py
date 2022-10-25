from typing import (
    Any,
    Dict,
    Tuple
)

from random import random

import numpy as np

import torch

from torchvision import transforms as T
from torchvision.transforms import functional as F


#TODO: Take more from Monai

class Normalize(object):
    """Normalizes the image from 0 to 1.

    Args:
        max_value (int): maximum reachable value
        return_bool (bool): if only 0 or 1 should be returned
    """

    def __init__(
        self,
        return_bool: bool = False,
        max_value: int = 65535,
    ) -> None:
        
        self.max_value = max_value
        self.return_bool = return_bool

    def __call__(self, image):
        image = image / self.max_value
        if(self.return_bool):
            image = (image > 0.1).float()
        return image


def shared_transform(
    random_crop: int = 0,
    affine: int = 0,
    horizontal_flip: int = 0,
    grayscale: int = 0,
) -> T.Compose:
    transforms = []

    transforms.append(T.ToTensor())
    rand_ints = np.random.rand(4)

    if rand_ints[0] <= affine:
        transforms.append(
            T.RandomAffine(
                degrees = (-60, 60),
                translate = (0, 0.8),
            )
        )

    if rand_ints[1] <= random_crop:
        transforms.append(
            T.RandomCrop(size = (128, 128))
        )

    if rand_ints[2] <= horizontal_flip:
        transforms.append(
            T.RandomHorizontalFlip(p = 1)
        )
    
    if rand_ints[3] <= grayscale:
        transforms.append(
            T.Grayscale()
        )

    return T.Compose(transforms)


def sample_transform(
    color_jitter: int = 0,
    blur: int = 0,
)-> T.Compose:

    transforms = []

    rand_ints = np.random.rand(2)


    if rand_ints[0] <= color_jitter:
        bright = np.random.rand()
        contrast = np.random.rand()
        transforms.append(
            T.ColorJitter(
              brightness=bright,
              contrast=contrast
            )
        )

    if rand_ints[1] <= blur:
        transforms.append(
            T.GaussianBlur(9)
        )

    return T.Compose(
        transforms
    )


def from_numpy(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)


def compose(config: Dict[str, Any]) -> T.Compose:
    transforms = []

    for name, args in config:
        if name == "Lambda":
            fn = args[0]
            fn = globals()[fn]
            transforms.append(
                T.Lambda(lambda x: fn(x)) # TODO: find fn in this module
            )
        else:
            transform = getattr(T, name)
            transforms.append(
                transform(*args)
            )

    return T.Compose(transforms)