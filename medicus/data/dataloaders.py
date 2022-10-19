from typing import Any

import torch

import nnunet


# Use the torch data loader by default
DataLoader = torch.data.utils.DataLoader


class nnUNetDataLoader:
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Any:
        pass