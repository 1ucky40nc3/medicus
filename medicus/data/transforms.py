from typing import (
    Any,
    Dict,
    Tuple
)

import numpy as np

import torch

from torchvision import transforms as T

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
    crop: bool = False,
    crop_size: Tuple = (50,50),
    center_crop: bool = True,
    center_crop_size: Tuple = (160,160),
    input_size: Tuple = (160, 160),
    rotate: bool = False,
    rotate_range: Tuple = (-30, 30),
    affine: bool = False,
    translation_range: Tuple = (0, 0.5),
    p: int = 0.2,
) -> T.Compose:
    transforms = [] # TODO: Actually use theese transforms ;)

    if affine or rotate:
        if rotate:
            transform_rotation_range = rotate_range
        else:
            transform_rotation_range = 0

        if affine:
            transform_affine_range = translation_range
        else:
            transform_affine_range = (0, 0)

        transforms.append(
            T.RandomAffine(
                degrees = transform_rotation_range,
                translate = transform_affine_range,
            )
        )

    if crop:
        transforms.append(
            T.RandomCrop(
                crop_size, 
                padding=(
                    crop_size[0] // 4, 
                    crop_size[1] // 4
                )
            )
        )
    
    if center_crop:
        return T.Compose([
            T.ToTensor(),
            T.CenterCrop(center_crop_size),
            T.RandomApply(transforms, p=p)                 
        ])
            
    if len(transforms) > 0:
        return T.Compose([
            T.ToTensor(),
            T.RandomApply(transforms, p=p)                 
        ])
    else:
        return T.Compose([
            T.ToTensor(),               
        ])


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