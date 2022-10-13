from typing import Tuple

import json
from pathlib import Path

import numpy as np
from numpy.core.numeric import NaN

from scipy.ndimage import center_of_mass

from PIL import Image

import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio

import matplotlib.pyplot as plt
from matplotlib.colors import (
    ListedColormap, 
    Normalize
)
from matplotlib.patches import Circle

from medicus.data.utils import (
    rescale_affine
)


# define HU windows
wdw_sbone = Normalize(vmin=-500, vmax=1300, clip=True)
wdw_hbone = Normalize(vmin=-200, vmax=1000, clip=True)

#########################
# Resample and reorient #


def reorient_to(
    img: nib.spatialimages.SpatialImage, 
    axcodes_to: Tuple[str] = ('P', 'I', 'R'), 
    verbose: bool = False
) -> nib.nifti1.Nifti1Image:
    """Reorients the nifti from its original orientation to another specified orientation
    
    Args:
        img (nibabel.spatialimages.SpatialImage): nibabel image
        axcodes_to (tuple[str]): a tuple of 3 characters specifying the desired orientation
    
    Returns:
        newimg (nib.nifti1.Nifti1Image): The reoriented nibabel image
    
    Note:
        Source: https://github.com/anjany/verse
    """
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    if verbose:
        print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return newimg


def resample_nib(
    img: nib.spatialimages.SpatialImage,
    voxel_spacing: Tuple[int] = (1, 1, 1),
    order: int = 3,
    verbose : bool = False):
    """Resamples the nifti from its original spacing to another specified spacing
    
    Args:
        img (nib.spatialimages.SpatialImage): nibabel image
        voxel_spacing (tuple[int]): a tuple of 3 integers specifying the desired new spacing
        order (int): the order of interpolation
    
    Returns:
        new_img: The resampled nibabel image

    Note:
        Source: https://github.com/anjany/verse 
    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
    ]).astype(int))
    new_aff = rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(
        img, 
        (new_shp, new_aff), 
        order=order, 
        cval=-1024)
    if verbose:
        print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def resample_mask_to(msk, to_img, verbose = False):
    """Resamples the nifti mask from its original spacing to a new spacing specified by its corresponding image
    
    Args:
        msk: The nibabel nifti mask to be resampled
        to_img: The nibabel image that acts as a template for resampling
    
    Returns:
        new_msk: The resampled nibabel mask
    
    Note:
        Source: https://github.com/anjany/verse 
    """
    to_img.header['bitpix'] = 8
    to_img.header['datatype'] = 2  # uint8
    new_msk = nib.processing.resample_from_to(msk, to_img, order=0)
    if verbose:
        print("[*] Mask resampled to image size:", new_msk.header.get_data_shape())
    return new_msk


def pad_and_crop(
    img: nib.spatialimages.SpatialImage,
    shape: Tuple[int],
    const_value: int = 0
) -> nib.nifti1.Nifti1Image:
    """Shapes an image by padding and center cropping
    
    Args:
        img (nib.spatialimages.SpatialImage): nibabel image
        shape (tuple[int]): a tuple of 2 integers specifying the desired new shape in x/y direction
        const_value (int): pads the image with const_value
    
    Returns:
        cropped_img (nib.spatialimages.SpatialImage): The nibabel image in the new shape

    Note:
        Source: https://github.com/anjany/verse 
    """

    # Get image shape, x,y,z -> x is y is z is

    z, x,y = img.shape
    crop_x, crop_y = shape
    pad_x = x
    pad_y = y
    #pad_z = z
    padded_img = img
    if (x < crop_x) or (y < crop_y):
        if (x < crop_x):
            pad_x = crop_x
        if (y < crop_y):
            pad_y = crop_y

        add_x0 = (pad_x - x) // 2
        add_x1 = pad_x - x - add_x0

        add_y0 = (pad_y - y) // 2
        add_y1 = pad_y - y - add_y0

        pixel_array = img.get_fdata().copy()
        image_pad = np.pad(
            pixel_array, 
            [
                (0     , 0     ),
                (add_x0, add_x1),
                (add_y0, add_y1),
            ], 
            mode='constant', 
            constant_values=const_value
        )

        padded_img = nib.Nifti1Image(image_pad, img.affine, img.header)
    
    start_x = pad_x//2 - (crop_x//2)
    start_y = pad_y//2 - (crop_y//2)

    cropped_img = padded_img.slicer[:,start_x : start_x + crop_x, start_y : start_y + crop_y] 
    return cropped_img


def nifti_to_png(
    img: nib.spatialimages.SpatialImage,
    target_dir: str,
    start_index: int = 0
) -> None:
    """Creates png image from nifti-file
    
    Args:
        img (nib.spatialimages.SpatialImage): nibabel image
        target_dir (str): directiory where the images will be saved
        start_index (int): first index in naming the files
    
    Note:
        Source: https://github.com/anjany/verse 
    """
    pixel_3d = np.array(img.get_fdata())
    for i, pixel_2d in enumerate(pixel_3d):
        img = Image.fromarray(pixel_2d).convert("L")
        img.save(f'{target_dir}file{i + start_index}.png')

