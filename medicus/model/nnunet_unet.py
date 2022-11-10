from typing import (
    Union,
    Optional
)

import torch
import torch.nn as nn

import nnunet
from nnunet.run import default_configuration as nnunet_default


class nnUNetUNet(nn.Module):
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
        super().__init__()

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
            init_data=False, 
            batch_size=batch_size,
            init_model=True,
            init_optim=False
        )        
        self.model = self.trainer.network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)