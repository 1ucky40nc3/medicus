from typing import Tuple
from typing import List
from typing import Optional
from typing import Callable

from PIL import Image

import numpy as np

import torch


from .utils import list_dataset_files
from .utils import set_seed


class SharedTransformImageDataset:
    def __init__(
        self, 
        sample_dir: str,
        target_dir: str,
        transform: Callable,
        target_transform: Callable,
        shared_transform: Callable,
        share_transform_random_seed: bool = True,
        return_untransformed_sample: bool = True
    ) -> None:
        samples_list, targets_list = list_dataset_files(
            sample_dir, target_dir)

        self.samples_list = samples_list
        self.targets_list = targets_list
        self.len = len(samples_list)

        self.transform = transform
        self.target_transform = target_transform
        self.shared_transform = shared_transform
        self.share_transform_random_seed = share_transform_random_seed
        self.return_untransformed_sample = return_untransformed_sample

    def __len__(self) -> int:
        return self.len

    def share_seed(self) -> bool:
        return self.share_transform_random_seed

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples_list[index]
        target = self.targets_list[index]

        sample = Image.open(sample).convert("L")
        target = Image.open(target).convert("L")

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



class UNetDataset:
    def __init__(
        self, 
        sample_dir: str,
        target_dir: str,
        transform: Callable,
        target_transform: Callable,
        shared_transform: Callable,
        share_transform_random_seed: bool = True,
        pat_dir: bool = False,
    ) -> None:
        #pat_dir -> True: für jeden Patienten existiert ein einzelner Unterordner, False: alle Dateien im gleichen Ordner
        if(pat_dir):
          samples_list, targets_list = list_dir_dataset_files(
              sample_dir, target_dir)
        else:
          samples_list, targets_list = list_dataset_files(
              sample_dir, target_dir)

        self.samples_list = samples_list
        self.targets_list = targets_list
        self.len = len(samples_list)

        self.pat_dir = pat_dir

        self.transform = transform
        self.target_transform = target_transform
        self.shared_transform = shared_transform
        self.share_transform_random_seed = share_transform_random_seed

    def __len__(self) -> int:
        return self.len

    def share_seed(self) -> bool:
        return self.share_transform_random_seed

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples_list[index]
        target = self.targets_list[index]

        sample = Image.open(sample)
        target = Image.open(target)


        if self.share_seed():
            seed = np.random.randint(2147483647)
            set_seed(seed)
                    
                
        sample = self.shared_transform(sample)
        input_sample = sample

        if self.share_seed:
            set_seed(seed)
        
        target = self.shared_transform(target)

        
        return input_sample/65535, sample/65535, (target>0.5).float()

