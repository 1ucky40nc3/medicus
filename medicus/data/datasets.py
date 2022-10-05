from typing import Tuple
from typing import Optional
from typing import Callable
from typing import Any


from pathlib import Path


import numpy as np
import nibabel as nib

from PIL import Image

import torch


import torch
import os 

from medicus.data.utils import list_dataset_files, list_dir_dataset_files, set_seed
from medicus.data.nib_utils import *

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
        pat_dir: bool = False,
        sample_format: str = ".png",
        target_format: str = ".png",
        **kwargs
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
        pat_dir: bool = False,
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

class NiftiImageDataset:
    """The NiftiImageDataset class.

    This describes a dataset that loads nifti-files from sample and target
    directories, resamples, reorientes and normalizes samples and targets
    and later returns the data points as numpy arrays.

    Note: 

    Attrs:
        sample_dir (str): The directory of the sample files.
        target_dir (str): The directory if the target files.
        pat_dir (bool):   States if files are in a seperate directory.
        reshape (bool):   States if images should get a new shape through padding and cropping.
        new_shape (Tuple):  States the new shape of the images.
        get_slices (Tuple): States if 2d numpy arrays should be returned.
                            If this is deactivated 3d numpy arrays will be returned.
        sizing (bool):      States if the images should be resampled for 1mm each pixel.
        automatic_padding_value (bool): States if the value for padding should be determined automatically.
        padding_value (int):    States the value for padding areas.
        reorientation: (Tuple): States how the images should be reorienated.
        normalize(bool):        States if the image values should be normalized between 0 and 1.

    """
    images = []
    masks = []

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        pat_dir: bool,
        reshape: bool,
        new_shape: Tuple = (160, 160),
        get_slices: bool = True,
        sizing: bool = True,
        automatic_padding_value: bool = True,
        padding_value: int = 0,
        reorientation: Tuple = ('I', 'P', 'R'),
        normalize: bool = True,
        ):

        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.new_shape = new_shape
        self.sizing = sizing
        self.pat_dir = pat_dir
        self.automatic_padding_value = automatic_padding_value
        self.padding_value = padding_value
        self.reorientation = reorientation
        self.normalize = normalize
        self.reshape = reshape
        
        if(pat_dir):
          samples_list, targets_list = list_dir_dataset_files(
              sample_dir = self.img_dir, target_dir = self.mask_dir, sample_format = ".gz")
          """          samples_list_nii, targets_list_nii = list_dir_dataset_files(
              sample_dir = self.img_dir, target_dir = self.mask_dir, sample_format = ".nii")"""
        else:
          samples_list, targets_list = list_dataset_files(
              sample_dir = self.img_dir, target_dir = self.mask_dir, sample_format = ".gz")
          """          samples_list_nii, targets_list_nii = list_dataset_files(
              sample_dir = self.img_dir, target_dir = self.mask_dir, sample_format = ".nii")"""

        #samples_list.extend(samples_list_nii)
        #targets_list.extend(targets_list_nii)

        for image_file, mask_file in zip(samples_list, targets_list):
          img = nib.load(image_file)
          mask = nib.load(mask_file)

          img_canon = nib.as_closest_canonical(img)
          mask_canon = nib.as_closest_canonical(mask)

          if(self.sizing):
            res_img = resample_nib(img_canon)
            res_mask = resample_mask_to(mask_canon, res_img)
          else:
            res_img = img_canon
            res_mask = mask_canon

          #reo_img = reorient_to(img_canon, self.reorientation)
          #reo_mask = reorient_to(mask_canon, self.reorientation)

          reo_img = res_img
          reo_mask = res_mask

          self.min_value = np.amin(np.array(reo_img.get_fdata()))
          self.max_value = np.amax(np.array(reo_img.get_fdata()))

          if (self.automatic_padding_value):
            self.padding_value = self.min_value

          if(self.reshape):
            pad_img = pad_and_crop(reo_img, self.new_shape, self.padding_value)
            pad_mask = pad_and_crop(reo_mask, self.new_shape, 0)
          else:
            pad_img = reo_img
            pad_mask = reo_mask

          voxel_array_img = np.array(pad_img.get_fdata())
          voxel_array_mask = np.array(pad_mask.get_fdata())

          if(normalize):
            voxel_array_img = (voxel_array_img - self.min_value)/(self.max_value-self.min_value)
            voxel_array_mask = voxel_array_mask/np.amax(voxel_array_mask)

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
        s = f'Dataset class with {self.__len__()} images, '

        if self.normalize:
          s = s + f"normalized, "

        if self.sizing:
          s = s + f"padded to {self.new_shape}, "

        s = s + f"reorientated to {self.reorientation}."
          
        return s

class NiftiImageDataset_old:
    """Dataset which converts Nifti-Files to 2D or 3D Numpy-Arrays
        
    Parameters:
    ----------
    img_dir:  Directory where the image nifti files are stored
    pat_dir:  each patient needs its own directory or not
    mask_dir: Directory where the mask nifti files are stored
              each patient needs its own directory (same name as in img_dir)
    new_shape: shape which the images will be cropped/padded
    get_slices: tells, if output should be bitmap (True) or voxelmap(False)
    sizing: tells, if output should be giving back with each pixel sized to 1mm or not

    
    Returns:
    ----------
    np array tuple of image and mask

    """
    files = []
    images = []
    masks = []
    min_value = 0
    max_value = 1

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        pat_dir: bool,
        new_shape: Tuple = (160, 160),
        get_slices: bool = True,
        sizing: bool = True,
        padding_value: int = 0):

        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.new_shape = new_shape
        self.sizing = sizing
        self.pat_dir = pat_dir
        
        if(self.pat_dir):
          for f in self.img_dir.iterdir():
            if f.is_dir():
              for n in f.iterdir():
                if not n.is_dir():
                  file_name, file_extension = os.path.splitext(n)

                  if (file_extension == ".gz" or file_extension == ".nii"):
                    mask_dir_file = os.path.join(self.mask_dir, os.path.basename(f) )+ "/"

                    for mask in Path(mask_dir_file).iterdir():
                      mask_name, mask_extension = os.path.splitext(mask)
                      if (mask_extension == ".gz" or mask_extension == ".nii"):
                        self.files.append(self.combine_files(n, mask))
        else:
          f = self.img_dir
          for n in f.iterdir():
            if not n.is_dir():
              file_name, file_extension = os.path.splitext(n)

              if (file_extension == ".gz" or file_extension == ".nii"):
                mask_dir_file = self.mask_dir

                for mask in Path(mask_dir_file).iterdir():
                  mask_name, mask_extension = os.path.splitext(mask)
                  if (mask_extension == ".gz" or mask_extension == ".nii"):
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

          if(self.sizing):
            res_img = resample_nib(img_canon)
            res_mask = resample_mask_to(mask_canon, res_img)
          else:
            res_img = img_canon
            res_mask = mask_canon

          pad_img = pad_and_crop(res_img, self.new_shape, padding_value)
          pad_mask = pad_and_crop(res_mask, self.new_shape, 0)

          new_orientation = ('I', 'P', 'R')

          reo_img = reorient_to(pad_img, new_orientation)
          reo_mask = reorient_to(pad_mask, new_orientation)

          voxel_array_img = np.array(reo_img.get_fdata())
          self.min_value = np.amin(voxel_array_img)
          self.max_value = np.amax(voxel_array_img)
          print("values:", self.min_value, self.max_value)
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