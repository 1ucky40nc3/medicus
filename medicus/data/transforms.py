from random import random
from typing import Tuple
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch

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
    random_crop: bool = False,
    affine: bool = False,
    color_jitter: bool = False,
    horizontal_flip: bool = False,
    p: int = 0.3,
) -> T.Compose:
    transforms = [T.ToTensor]

    if affine:
        transforms.append(
            T.RandomAffine(
                degrees = (-60, 60),
                translate = (0, 0.8),
            )
        )

    if color_jitter:
        bright = torch.rand()
        contrast = torch.rand() 
        transforms.append(
            T.ColorJitter(brightness = bright,
            contrast = contrast)
        )

    if random_crop:
        transforms.append(
            T.RandomCrop(size = (100, 100))
        )

    if horizontal_flip:
        transforms.append(
            T.RandomHorizontalFlip(p = 0.5)
        )
    
            

    return T.Compose([
        T.ToTensor(),
        T.RandomApply(transforms, p = p)                 
    ])
