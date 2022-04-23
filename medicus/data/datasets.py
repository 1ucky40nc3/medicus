from typing import Tuple
from typing import Optional
from typing import Callable
from typing import Any

from pathlib import Path
from PIL import Image

import numpy as np
import nibabel as nib


import torch
import os 

from medicus.data.utils import list_dataset_files, list_dir_dataset_files, set_seed
from medicus.data.nib_utils import *

def identity(x: Any):
    return x


class SharedTransformImageDataset:
    def __init__(
        self, 
        sample_dir: str,
        target_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shared_transform: Optional[Callable] = None,
        share_transform_random_seed: bool = True,
        return_untransformed_sample: bool = True
    ) -> None:
        samples_list, targets_list = list_dataset_files(
            sample_dir, target_dir)

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

class NiftiImageDataset:
    """Dataset which converts Nifti-Files to 2D Numpy-Arrays
        
    Parameters:
    ----------
    img_dir:  Directory where the image nifti files are stored
              each patient needs its own directory
    mask_dir: Directory where the mask nifti files are stored
              each patient needs its own directory (same name as in img_dir)
    new_shape: shape which the images will be cropped/padded
    get_slices: tells, if output should be bitmap (True) or voxelmap(False)

    
    Returns:
    ----------
    np array tuple of image and mask

    """
    files = []
    images = []
    masks = []

    def __init__(self, img_dir, mask_dir, new_shape = (160, 160), get_slices = True):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.new_shape = new_shape
        for f in self.img_dir.iterdir():
          if f.is_dir():
            for n in f.iterdir():
              if not n.is_dir():
                file_name, file_extension = os.path.splitext(n)

                if (file_extension == ".gz"):
                  mask_dir_file = os.path.join(self.mask_dir, os.path.basename(f) )+ "/"

                  for mask in Path(mask_dir_file).iterdir():
                    mask_name, mask_extension = os.path.splitext(mask)
                    if (mask_extension == ".gz"):
                      self.files.append(self.combine_files(n, mask))

        len_files = len(self.files)
        for i in range(len_files):
          try:
            file = self.files[i]
          except:
            break
          img = nib.load(file['image'])
          mask = nib.load(file['mask'])

          img_canon = nib.as_closest_canonical(img)
          mask_canon = nib.as_closest_canonical(mask)

          res_img = resample_nib(img_canon)
          res_mask = resample_mask_to(mask_canon, res_img)


          pad_img = pad_and_crop(res_img, self.new_shape, -1024)
          pad_mask = pad_and_crop(res_mask, self.new_shape, 0)

          new_orientation = ('I', 'P', 'R')

          reo_img = reorient_to(pad_img, new_orientation)
          reo_mask = reorient_to(pad_mask, new_orientation)

          voxel_array_img = np.array(reo_img.get_fdata())
          voxel_array_mask = np.array(reo_mask.get_fdata())

          if get_slices:
            for pixel_array in voxel_array_img:
              self.images.append(pixel_array)

            for pixel_array in voxel_array_mask:
              self.masks.append(pixel_array)
          else:
            self.images.append(voxel_array_img)
            self.masks.append(voxel_array_mask)
    
    def combine_files(self, image, mask,): 
        return {
            'image': image, 
            'mask': mask,
        }

    def __len__(self):
        img_len = len(self.images)

        return img_len


    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

    def __repr__(self):
        s = f'Dataset class with {self.__len__()} images, padded to {self.new_shape}'

        return s

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

