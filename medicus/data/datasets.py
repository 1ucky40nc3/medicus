from typing import Tuple
from typing import Optional
from typing import Callable
from typing import Any


import numpy as np

from PIL import Image

import torch

from .utils import list_dataset_files
from .utils import set_seed


def identity(x: Any):
    return x



class SharedTransformDataset:
    """The abstract SharedTransformDataset class.

    This describes a dataset that loads files from sample and target
    directories, computes sample, target and even shared transforms
    and later returns the data points as tuples.

    Note: 
        1) If shared transforms between samples and targets aren't needed
           than it is advised to use datasets such as 
           `torchvision.datasets.ImageFolder` (for images) instead.
        2) The `self.load` function has the be implemented as this is 
           an abstract class, to load the desired data from different file types.

    Attrs:
        sample_dir (str): The directory of the sample files.
        target_dir (str): The directory if the target files.
        transform (call): Sample transforms (see `torchvision.transforms`).
        target_transform (call): Target transforms (see `torchvision.transforms`).
        shared_transform (call): Shared transforms for sample and target
                                 (see `torchvision.transforms`).
        share_transform_random_seed (bool): States if the shared transforms shall
                                            share their seed. If this deactivated
                                            than other datasets such as
                                            `torchvision.datasets.ImageFolder`
                                            (for images) should be used.
        return_untransformed_sample (bool): States if instead if a (input, target)
                                            tuple a (input, sample, target) tuple
                                            shall be returned. Here an extra
                                            untransformed sample is also returned.
    """
    def __init__(
        self, 
        sample_dir: str,
        target_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shared_transform: Optional[Callable] = None,
        share_transform_random_seed: bool = True,
        return_untransformed_sample: bool = True,
        sample_format: str = ".png",
        target_format: str = ".png",
        **kwargs
    ) -> None:
        samples_list, targets_list = list_dataset_files(
            sample_dir, target_dir, sample_format, target_format)

        self.samples_list = samples_list
        self.targets_list = targets_list
        self.len = len(samples_list)

        self.transform = transform if transform else identity
        self.target_transform = target_transform if target_transform else identity
        self.shared_transform = shared_transform if shared_transform else identity
        self.share_transform_random_seed = share_transform_random_seed
        self.return_untransformed_sample = return_untransformed_sample

    def __len__(self) -> int:
        return self.len

    def share_seed(self) -> bool:
        return self.share_transform_random_seed

    def load(self, path: str) -> Any:
        raise NotImplementedError

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples_list[index]
        target = self.targets_list[index]

        sample = self.load(sample)
        target = self.load(target)

        if self.share_seed():
            seed = np.random.randint(2147483647)
            set_seed(seed)
                    
        sample = self.shared_transform(sample)
        input = self.transform(sample)

        if self.share_seed:
            set_seed(seed)
        
        target = self.shared_transform(target)
        target = self.target_transform(target)
        
        if self.return_untransformed_sample:
            return input, sample, target
        return input, target


class SharedTransformImageDataset(SharedTransformDataset):
    """The SharedTransformImageDataset class.

    This describes a dataset that loads files from sample and target
    directories, computes sample, target and even shared transforms
    and later returns the data points as tuples.

    Note: 
        1) If shared transforms between samples and targets aren't needed
           than it is advised to use datasets such as 
           `torchvision.datasets.ImageFolder` instead.

    Attrs:
        sample_dir (str): The directory of the sample files.
        target_dir (str): The directory if the target files.
        transform (call): Sample transforms (see `torchvision.transforms`).
        target_transform (call): Target transforms (see `torchvision.transforms`).
        shared_transform (call): Shared transforms for sample and target
                                (see `torchvision.transforms`).
        share_transform_random_seed (bool): States if the shared transforms shall
                                            share their seed. If this deactivated
                                            than other datasets such as
                                            `torchvision.datasets.ImageFolder`
                                            should be used.
        return_untransformed_sample (bool): States if instead if a (input, target)
                                            tuple a (input, sample, target) tuple
                                            shall be returned. Here an extra
                                            untransformed sample is also returned.
    """
    def __init__(
        self, 
        sample_dir: str,
        target_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shared_transform: Optional[Callable] = None,
        share_transform_random_seed: bool = True,
        return_untransformed_sample: bool = True,
        sample_format: str = ".png",
        target_format: str = ".png",
        **kwargs
    ) -> None:
        args = locals()
        del args["self"]
        del args["__class__"]
        super().__init__(**args)
        
    def load(self, path: str) -> Image:
        return Image.open(path).convert("RGB")


class SharedTransformNumpyDataset(SharedTransformDataset):
    """The SharedTransformNumpyDataset class.

    This describes a dataset that loads files from sample and target
    directories, computes sample, target and even shared transforms
    and later returns the data points as tuples.

    Note: 
        1) If shared transforms between samples and targets aren't needed
           than it is advised to use different dataset classes.
        2) The samples and targets should have the basic shape of [..., H, W, C].
           This is ment to encourage the compatibility the conversion transforms
           of `torchvision.transform`. Yet this is not needed and can be configured
           freely. ;)

    Attrs:
        sample_dir (str): The directory of the sample files.
        target_dir (str): The directory if the target files.
        transform (call): Sample transforms (see `torchvision.transforms`).
        target_transform (call): Target transforms (see `torchvision.transforms`).
        shared_transform (call): Shared transforms for sample and target
                                (see `torchvision.transforms`).
        share_transform_random_seed (bool): States if the shared transforms shall
                                            share their seed. If this deactivated
                                            than other datasets such as
                                            `torchvision.datasets.ImageFolder`
                                            should be used.
        return_untransformed_sample (bool): States if instead if a (input, target)
                                            tuple a (input, sample, target) tuple
                                            shall be returned. Here an extra
                                            untransformed sample is also returned.
    """
    def __init__(
        self, 
        sample_dir: str,
        target_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shared_transform: Optional[Callable] = None,
        share_transform_random_seed: bool = True,
        return_untransformed_sample: bool = True,
        sample_format: str = ".npy",
        target_format: str = ".npy",
        **kwargs
    ) -> None:
        args = locals()
        del args["self"]
        del args["__class__"]
        super().__init__(**args)
    
    def load(self, path: str) -> np.ndarray:
        return np.load(path)