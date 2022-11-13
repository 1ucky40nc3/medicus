from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def softdiceloss(
    output: torch.Tensor,
    target: torch.Tensor,
    nonlinearity: Callable = torch.sigmoid,
    smooth: float = 1.
) -> torch.Tensor:
    """A soft dice loss implementation.
    Note:
        - Adapted from: https://github.com/usuyama/pytorch-unet/blob/master/loss.py
    """
    output = nonlinearity(output)

    A = output.contiguous()
    B = target.contiguous()

    # compute dice per element in batch
    intersection = (A * B).sum(dim=2).sum(dim=2)
    A = A.sum(dim=2).sum(dim=2)
    B = B.sum(dim=2).sum(dim=2)

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
    bce_weight: float = .5,
    nonlinearity: Callable = torch.sigmoid,
    smooth: float = 1.
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        input=output, 
        target=target
    )
    dice = softdiceloss(
        output=output, 
        target=target,
        nonlinearity=nonlinearity,
        smooth=smooth
    )
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


class BinaryCrossEntropyAndSoftDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = .5,
        nonlinearity: Callable = torch.sigmoid,
        smooth: float = 1.
    ) -> None:
        self.bce_weight = bce_weight
        self.nonlinearity = nonlinearity
        self.smooth = smooth

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = bce_and_softdiceloss(
            output=output,
            target=target,
            bce_weight=self.bce_weight,
            nonlinearity=self.nonlinearity,
            smooth=self.smooth
        )
        return loss

    