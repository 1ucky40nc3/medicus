import argparse
import time
import torch
import torch.nn as nn

from typing import Tuple
from typing import List
from typing import Optional
from typing import Callable
from typing import Any
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm
from torch.utils import tensorboard
import wandb
import torchvision.transforms.functional as F
import numpy as np
from colour import Color


Device = Any


def timestamp() -> str:
    return time.strftime(
        "%Y%m%d%H%M%S", 
        time.localtime()
    )


def masks_to_colorimg(masks):
    colors = np.asarray([
        (201, 58, 64), 
        (242, 207, 1), 
        (0, 152, 75), 
        (101, 172, 228), 
        (56, 34, 132), 
        (160, 194, 56)
    ])

    # shape: [H, W, 3]
    colorimg = np.ones(
        (masks.shape[1], masks.shape[2], 3), 
        dtype=np.float32
    ) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)
  

def masks2imgs(masks):
    masks = masks.cpu().numpy()
    batch, channels, height, width = masks.shape

    red, blue = Color("red"), Color("blue")
    colors = list(red.range_to(blue, channels))
    colors = np.array([c.rgb for c in colors]) * 255

    imgs = np.ones(
        (batch, height, width, 3), 
        dtype=np.float32
    ) * 255

    for i in range(batch):
        for y in range(height):
            for x in range(width):
                selected_colors = colors[masks[i, :, y, x] > 0.5]

                if len(selected_colors) > 0:
                    imgs[i, y, x, :] = np.mean(selected_colors, axis=0)

    imgs = imgs.transpose((0, 3, 1, 2))
    imgs = torch.from_numpy(imgs).contiguous()
    imgs = imgs.float().div(255)

    return imgs


class ArgumentParser:
    def __init__(self, *args) -> None:
        self.parser = argparse.ArgumentParser(parents=args)
    
    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


def parse(args: argparse.Namespace, **kwargs) -> dict:
    config = {**vars(args), **kwargs}
    return config