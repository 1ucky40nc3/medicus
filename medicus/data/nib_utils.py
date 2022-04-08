from pathlib import Path
from numpy.core.numeric import NaN
import numpy as np
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Circle
import json
import medicus.utils.affines as ua

# define HU windows
wdw_sbone = Normalize(vmin=-500, vmax=1300, clip=True)
wdw_hbone = Normalize(vmin=-200, vmax=1000, clip=True)

#########################
# Resample and reorient #


def reorient_to(img, axcodes_to=('P', 'I', 'R'), verb=False):
    """Reorients the nifti from its original orientation to another specified orientation
    
    Parameters:
    ----------
    img: nibabel image
    axcodes_to: a tuple of 3 characters specifying the desired orientation
    
    Returns:
    ----------
    newimg: The reoriented nibabel image 
    
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
    if verb:
        print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return newimg


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3, verb = False):
    """Resamples the nifti from its original spacing to another specified spacing
    
    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation
    
    Returns:
    ----------
    new_img: The resampled nibabel image 
    
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
    new_aff = ua.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    if verb:
        print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def resample_mask_to(msk, to_img, verb = False):
    """Resamples the nifti mask from its original spacing to a new spacing specified by its corresponding image
    
    Parameters:
    ----------
    msk: The nibabel nifti mask to be resampled
    to_img: The nibabel image that acts as a template for resampling
    
    Returns:
    ----------
    new_msk: The resampled nibabel mask 
    
    """
    to_img.header['bitpix'] = 8
    to_img.header['datatype'] = 2  # uint8
    new_msk = nib.processing.resample_from_to(msk, to_img, order=0)
    if verb:
        print("[*] Mask resampled to image size:", new_msk.header.get_data_shape())
    return new_msk


def pad_and_crop(img, shape, const_value = 0):
    """Shapes an image by padding and center cropping
    
    Parameters:
    ----------
    img: nibabel image
    shape: a tuple of 2 integers specifying the desired new shape in x/y direction
    const_value: pads the image with const_value
    
    Returns:
    ----------
    cropped_img: The nibabel image in the new shape
    
    """

    # Get image shape, x,y,z -> x is y is z is

    x,y,z = img.shape
    crop_x, crop_y, crop_z = shape
    pad_x = x
    pad_y = y
    pad_z = z
    padded_img = img
    if (x < crop_x) or (y < crop_y) or (z < crop_z):
      if (x < crop_x):
        pad_x = crop_x
      if (y < crop_y):
        pad_y = crop_y
      if (z < crop_z):
        pad_z = crop_z

      add_x0 = (pad_x - x)//2
      add_x1 = pad_x - x - add_x0

      add_y0 = (pad_y - y)//2
      add_y1 = pad_y - y - add_y0
        
      add_z0 = (pad_z - z)//2
      add_z1 = pad_z - z - add_z0

      pixel_array = img.get_fdata().copy()
      image_pad = np.pad(pixel_array, [(add_x0,add_x1),(add_z0,add_z1),(add_y0,add_y1)], mode = 'constant', constant_values = const_value)

      padded_img = nib.Nifti1Image(image_pad, img.affine, img.header)
    
    start_x = pad_x//2 - (crop_x//2)
    start_y = pad_y//2 - (crop_y//2)
    start_z = pad_z//2 - (crop_z//2)



    cropped_img = padded_img.slicer[start_x : start_x + crop_x, start_y : start_y + crop_y, start_z : start_z + crop_z] 
    return cropped_img


def nifti_to_png(img, target_dir, start_index = 0):
    """Creates png image from nifti-file
    
    Parameters:
    ----------
    img: nibabel image
    target_dir: directiory where the images will be saved
    start_index: first index in naming the files

    Returns:
    ----------
    
    """
    pixel_3d = np.array(img.get_fdata())
    for i, pixel_2d in enumerate(pixel_3d):
        img = Image.fromarray(pixel_2d).convert("L")
        img.save(f'{target_dir}file{i + start_index}.png')

