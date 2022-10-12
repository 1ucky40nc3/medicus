from typing import Any
from typing import Dict
from typing import Tuple
from typing import Callable
from typing import Optional
from typing import List

import os
import sys
import math
import json
import logging

from tqdm import tqdm

import numpy as np

import torch.nn as nn
from torch.utils import tensorboard
from torchmetrics import MeanMetric
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary
from colour import Color

import torchvision.transforms.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as TNF

import torchvision

import time
import copy

from collections import defaultdict

import wandb

Device = Any
LRScheduler = Any

from .medicus.objectives.unet import dice_loss
from .medicus.utils import timestamp, parse, inference, evaluate, Writer, masks2imgs


def inference(
    model: nn.Module,
    samples: torch.Tensor,
    device: Device
) -> Tuple[torch.Tensor]:
    with torch.no_grad():
        samples = samples.to(device)
        outputs = model(samples)

    return outputs