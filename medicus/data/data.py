from typing import (
    Iterable,
)

import medicus


def load_data(
    cfg: dict,
    split: str = "train"
) -> Iterable:
    ds_cfg = cfg["dataset"]
    dl_cfg = cfg["dataloader"]

    dataset_cls =  medicus.utils.get_cls(
        medicus.data.datasets, 
        ds_cfg["name"]
    )
    transforms = medicus.data.transforms.compose(ds_cfg["transforms"])

    dataset = None or dataset_cls(
        transforms=transforms,
        **ds_cfg["config"],
    )

    dataloader_cls = medicus.utils.get_cls(
        medicus.data.dataloaders, 
        dl_cfg["name"]
    )

    dataloader = dataloader_cls(
        dataset=dataset,
        shuffle=(split == "train"),
        **dl_cfg["config"],
    )

    return dataloader

