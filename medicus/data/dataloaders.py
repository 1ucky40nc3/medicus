from typing import (
    Iterator,
    Tuple,
    Union
)

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
        **kwargs
    ) -> None:
        default = nnunet_default.get_default_configuration(
            network, task, network_trainer, plans_identifier)

        trainer_class = default[5]
        trainer = trainer_class(
            plans_file=default[0],
            fold=fold,
            output_folder=default[1],
            dataset_directory=default[2],
            batch_dice=default[3],
            stage=default[4],
            unpack_data=unpack_data,
            deterministic=deterministic,
            fp16=fp16
        )
        trainer.initialize(not validation_only)
        self.gen = trainer.tr_gen if split == "train" else trainer.val_gen

    def __iter__(self) -> Iterator:
        return self

    
    def __next__(self) -> Tuple[torch.Tensor]:
        data = self.gen.next()
        return data["data"], data["target"]