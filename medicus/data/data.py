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

    ds_cfg = medicus.utils.load_cfg(cfg=ds_cfg, split=split)
    dl_cfg = medicus.utils.load_cfg(cfg=dl_cfg, split=split)

    dataset_cls =  medicus.utils.get_cls(
        medicus.data.datasets, 
        ds_cfg["name"]
    )
    transforms = medicus.data.transforms.compose(ds_cfg["transforms"])

    dataset = None or dataset_cls(
        transforms=transforms,
        **ds_cfg.get("config", {}),
    )

    dataloader_cls = medicus.utils.get_cls(
        medicus.data.data_loaders, 
        dl_cfg["name"]
    )

    dataloader = dataloader_cls(
        dataset=dataset,
        shuffle=(split == "train"),
        **dl_cfg.get("config", {}),
    )

    return dataloader