from typing import Tuple
from typing import List
from typing import Optional
from typing import Callable

import os
import glob
import random
import torch


def filename(path: str) -> str:
    filename = path.split(os.sep)[-1]
    filename = filename.split(".")[:-1]
    return ".".join(filename)


def filenames(paths: str) -> List[str]:
    return list(map(filename, paths))


def list_dataset_files(
    sample_dir: str, 
    target_dir: str,
    sample_format: str=".png",
    target_format: str=".png"
) -> Tuple[List[str], List[str]]:
    samples_list = glob.glob(f"{sample_dir}/*{sample_format}")
    targets_list = glob.glob(f"{target_dir}/*{target_format}")

    samples_list = list(sorted(samples_list))
    targets_list = list(sorted(targets_list))

    assert len(samples_list) > 0, "ERROR: No samples were found!"
    assert len(targets_list) > 0, "ERROR: No targets were found!"
    assert len(samples_list) == len(targets_list), "ERROR: Different number sample and target files!"
    assert filenames(samples_list) == filenames(targets_list), "ERROR: Sample and target filenames don't match!"

    print(f"SUCCESS: A total of {len(samples_list)} samples were found!")
    return samples_list, targets_list


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)