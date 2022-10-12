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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: Device,
    log_every: int = 50,
    desc: str = "Evaluating...",
    tqdm_config: dict = {}
) -> Any:
    model.eval()

    metric = MeanMetric()
    with torch.no_grad():
        with tqdm(dataloader, desc=desc, unit="batch", **tqdm_config) as iterator:
            for i, (x, y) in enumerate(iterator):
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                loss = loss_fn(outputs, y)
                metric.update(loss.cpu())

                if (i + 1) % log_every == 0:
                    iterator.set_postfix(
                        mean_loss=metric.compute().item())

    return metric.compute()