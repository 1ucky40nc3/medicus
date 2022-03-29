from ast import Call
from typing import List
from typing import Callable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# see: #https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
def tsum(
    tensor: torch.Tensor,
) -> torch.Tensor:
    axes = [0] + list(range(2, tensor.shape))
    axes = np.unique(axes).astype(int)

    for ax in sorted(axes, reverse=True):
        tensor = tensor.sum(dim=ax)
    
    return tensor


# see: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
def nnunet_softdiceloss(
    output: torch.Tensor,
    target: torch.Tensor,
    nonlinearity: Callable = torch.sigmoid,
    smooth: float = 1e-5,
    eps: float = 1e-8
) -> torch.Tensor:
    output = nonlinearity(output)

    tp = output * target
    fp = output * (1 - target)
    fn = (1 - output) * target

    # compute values as single value across batch
    tp = tsum(tp)
    fp = tsum(fp)
    fn = tsum(fn)

    nominator = 2 * tp + smooth
    denominator = 2 * tp + fp + fn + smooth

    loss = nominator / (denominator + eps)
    return -loss


def nnunet_softdiceloss_refactor(
    output: torch.Tensor,
    target: torch.Tensor,
    nonlinearity: Callable = torch.sigmoid,
    smooth: float = 1e-5,
    eps: float = 1e-8
) -> torch.Tensor:
    output = nonlinearity(output)

    A = output
    B = target

    intersection = output * target

    # compute values as single value across batch
    intersection = tsum(intersection)
    A = tsum(A)
    B = tsum(B)

    nominator = 2 * intersection + smooth
    denominator = A + B + smooth

    loss = nominator / denominator
    return -loss


# see: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
def bce_and_nnunet_softdiceloss(
    output: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.,
    dice_weight: float = 1.
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        output, target)
    dice = nnunet_softdiceloss(output, target)
    loss = bce * bce_weight + dice (1 - dice_weight)

    return loss


# see: https://github.com/usuyama/pytorch-unet/blob/master/loss.py
def softdiceloss(
    output: torch.Tensor,
    target: torch.Tensor,
    nonlinearity: Callable = F.sigmoid,
    smooth: float = 1.
) -> torch.Tensor:
    output = nonlinearity(output)

    A = output.contiguous()
    B = target.contiguous()

    # compute dice per element in batch
    intersection = (A * B).sum(dim=2).sum(dim=2)
    A = A.sum(dim=2).sum(dim=2)
    b = B.sum(dim=2).sum(dim=2)

    nominator = 2. * intersection + smooth
    denominator = A + B + smooth

    loss = 1 - nominator / denominator
    # average loss over batch
    loss = loss.mean()

    return loss


# see: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.ipynb
def bce_and_softdiceloss(
    output: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        output, target)
    dice = softdiceloss(output, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


