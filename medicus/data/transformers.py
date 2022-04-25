from typing import Tuple
from torchvision import transforms as T

#TODO: TAke more from Monai

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
        ):
        
        self.max_value = max_value
        self.return_bool = return_bool

    def __call__(self, image):
        image = image/self.max_value
        if(self.return_bool):
            image = (image > 0.5).float()
        return image

def shared_transform(
    crop: bool = True,
    crop_size: Tuple = (50,50),
    input_size: Tuple = (160, 160),
    rotate: bool = True,
    rotate_range: Tuple = (-30, 30),
    affine: bool = True,
    translation_range: Tuple = (0, 0.5),
    p: int = 0.2,
    ):

    transformers = []

    if(affine or rotate):
        if(rotate):
            transform_rotation_range = rotate_range
        else:
            transform_rotation_range = 0

        if(affine):
            transform_affine_range = translation_range
        else:
            transform_affine_range = (0,0)

        transformers.append(
            T.RandomAffine(
                degrees = transform_rotation_range,
                translate = transform_affine_range,
            )
        )

    if(crop):
        transformers.append(
            T.RandomCrop(crop_size, padding = (crop_size[0]//4, crop_size[1]//4))
        )
    



    if(len(transformers) > 0):
        return T.Compose([
        T.ToTensor(),
        T.RandomApply(transformers, p = p)                 
        ])
    else:
        return T.Compose([
        T.ToTensor(),               
        ])

sample_transform = T.Compose([
    Normalize(),
])

target_transform = T.Compose([
    Normalize(return_bool = True),
])
