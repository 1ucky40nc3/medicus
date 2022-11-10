from typing import (
    Optional,
    Iterator,
    Tuple,
    Union
)

import math
import json

import numpy as np

import torch

import nnunet
from nnunet.run import default_configuration as nnunet_default


# Use the torch data loader by default
DataLoader = torch.utils.data.DataLoader


class nnUNetDataLoader:
    def __init__(
        self,
        task: str,
        fold: Union[str, int] = "all",
        network: str = "2d",
        network_trainer: str = "nnUNetTrainerV2",
        validation_only: bool = False,
        plans_identifier: str = "nnUNetPlansv2.1",
        unpack_data: bool = True,
        deterministic: bool = False,
        fp16: bool = False,
        split: str = "train",
        batch_size: Optional[int] = None,
        pin_memory: bool = False,
        convert_to_tensor: bool = False,
        **kwargs
    ) -> None:
        self.task = task
        self.fold = fold
        self.network = network
        self.network_trainer = network_trainer
        self.validation_only = validation_only
        self.plans_identifier = plans_identifier
        self.unpack_data = unpack_data
        self.deterministic = deterministic
        self.fp16 = fp16
        self.split = split
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.convert_to_tensor = convert_to_tensor

        default = nnunet_default.get_default_configuration(
            network, task, network_trainer, plans_identifier)

        self.plans_file = default[0]
        self.output_folder = default[1]
        self.dataset_directory = default[2]
        self.batch_dice = default[3]
        self.stage = default[4]
        self.trainer_class = default[5]

        self.trainer = self.trainer_class(
            plans_file=self.plans_file,
            fold=fold,
            output_folder=self.output_folder,
            dataset_directory=self.dataset_directory,
            batch_dice=self.batch_dice,
            stage=self.stage,
            unpack_data=unpack_data,
            deterministic=deterministic,
            fp16=fp16,
            pin_memory=pin_memory,
            convert_to_tensor=convert_to_tensor
        )
        self.trainer.initialize(
            training=not validation_only, 
            init_data=True, 
            batch_size=batch_size,
            init_model=False,
            init_optim=False
        )
        self.gen = self.trainer.tr_gen if split == "train" else self.trainer.val_gen

        self.dataset_config = json.load(open(f"{self.dataset_directory}/dataset.json"))
        self.num_samples = len(self.dataset_config["training"])

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Tuple[torch.Tensor]:
        data = self.gen.next()
        sample = data["data"]
        target = data["target"]

        if isinstance(target, list):
            target = target[0]

        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        return sample, target

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.batch_size)