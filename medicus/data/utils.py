from typing import Tuple
from typing import List
from typing import Optional
from typing import Callable
from pathlib import Path
from PIL import Image

import os
import glob
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from collections import deque


def filename(path: str) -> str:
    filename = path.split(os.sep)[-1]
    filename = filename.split(".")[:-1]
    return ".".join(filename)


def filenames(paths: str) -> List[str]:
    return list(map(filename, paths))


def list_dataset_files(
    sample_dir: str, 
    target_dir: str,
    sample_format: str=".*",
    target_format: str=".*",
    allow_different_names = False,
) -> Tuple[List[str], List[str]]:
    samples_list = glob.glob(f"{sample_dir}/*{sample_format}")
    targets_list = glob.glob(f"{target_dir}/*{target_format}")

    samples_list = list(natsorted(samples_list))
    targets_list = list(natsorted(targets_list))

    assert len(samples_list) > 0, "ERROR: No samples were found!"
    assert len(targets_list) > 0, "ERROR: No targets were found!"
    assert len(samples_list) == len(targets_list), "ERROR: Different number sample and target files!"
    if not allow_different_names:
        assert filenames(samples_list) == filenames(targets_list), "ERROR: Sample and target filenames don't match!"

    print(f"SUCCESS: A total of {len(samples_list)} samples were found!")
    return samples_list, targets_list


def list_dir_dataset_files(
    sample_dir: str, 
    target_dir: str,
    sample_format: str=".png",
    target_format: str=".png",
    allow_different_names = False
  ) -> Tuple[List[str], List[str]]:
    sample_dirs = [dir for dir in Path(sample_dir).iterdir()]
    target_dirs = [dir for dir in Path(target_dir).iterdir()]
    samples_list = []
    targets_list = []
    for s_dir in sample_dirs:
      samples_list.extend(glob.glob(f"{s_dir}/*{sample_format}"))
    for t_dir in target_dirs:
      targets_list.extend(glob.glob(f"{t_dir}/*{target_format}"))

    samples_list = list(natsorted(samples_list))
    targets_list = list(natsorted(targets_list))


    print(len(samples_list),len(targets_list))
    assert len(samples_list) > 0, "ERROR: No samples were found!"
    assert len(targets_list) > 0, "ERROR: No targets were found!"
    assert len(samples_list) == len(targets_list), "ERROR: Different number sample and target files!"
    if not allow_different_names:
        assert filenames(samples_list) == filenames(targets_list), "ERROR: Sample and target filenames don't match!"

    print(f"SUCCESS: A total of {len(samples_list)} samples were found!")
    return samples_list, targets_list


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    
def batch_to_img(img, mask, comb = True):
    """
    Funktion zur Darstellung eines Batches
    ---
    input:
        img: Bild 5d-Array mit batchsize, channels, und 2d Bild
        mask: Masken 5d-Array mit batchsize, channels, und 2d Maske
        comb: Gibt an, ob ein Overlay-Bild erzeugt werden soll
    
    """
    batch_size = img.shape[0]

    if comb:
        fig, ax = plt.subplots(batch_size, 3, figsize=(15, batch_size * 5))
        for i in range(batch_size):
            x = img[i]
            y = mask[i]
            comb = torch.cat((x, x, y), dim = 0)

            ax[i, 0].imshow(x[0])
            ax[i, 1].imshow(y[0])
            ax[i, 2].imshow(np.dstack(comb))
    else:
        fig, ax = plt.subplots(batch_size,2, figsize=(15,batch_size*5))
        for i in range(batch_size):
            x = img[i]
            y = mask[i]

            ax[i, 0].imshow(x[0])
            ax[i, 1].imshow(y[0])

    plt.show()    

def batch_to_pred(model, img, mask, comb = True):
  """
  Funktion zur Darstellung eines Batches und optischer Evaluation eines Models
  ---
  input:
      model: Model, dessen Vorhersage für img gezeigt werden soll
      img: Bild 5d-Array mit batchsize, channels, und 2d Bild
      mask: Masken 5d-Array mit batchsize, channels, und 2d Maske
      comb: Gibt an, ob ein Overlay-Bild erzeugt werden soll
    
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  batch_size = img.shape[0]
  inputs = img.to(device)

  pred = model(inputs)
  pred1 = torch.sigmoid(pred.cpu()) 

  if comb:
    fig, ax = plt.subplots(batch_size, 5, figsize=(15, batch_size*5))
    for i in range(batch_size):
      x = img[i]
      y = mask[i]
      z = pred1[i]
      comb_mask = torch.cat((x, x, y), dim = 0)
      comb_pred = torch.cat((x, x, z), dim = 0)

      ax[i, 0].imshow(x[0])
      ax[i, 1].imshow(y[0])
      ax[i, 2].imshow(np.dstack(comb_mask))
      ax[i, 3].imshow(z[0].detach().numpy())
      ax[i, 4].imshow(np.dstack(comb_pred.detach().numpy()))

  else:
    fig, ax = plt.subplots(batch_size, 3, figsize=(15, batch_size*5))
    for i in range(batch_size):
      x = img[i]
      y = mask[i]
      z = pred1[i]

      ax[i, 0].imshow(x[0])
      ax[i, 1].imshow(y[0])
      ax[i, 2].imshow(z[0])
  plt.show()
    
  
def save_data_as_png(target_dir, data, start_with = 0, addition = 1024, mult = 65535):
    for i, (x, y) in enumerate(data):
        x = (x + addition)*mult#/4095
        y = y * mult
        img = Image.fromarray(x).convert('I')
        mask = Image.fromarray(y).convert('I')
        img.save(f'{target_dir}/images/file{i + start_with}.png')
        mask.save(f'{target_dir}/masks/file{i + start_with}.png')

def save_voxel_as_png(target_dir, data, addition = 0, mult = 1, isDataloader = True):
    if isDataloader:
        for i, (x, y) in enumerate(data):
            img_path = target_dir + f'/images/pat{i}'
            mask_path = target_dir + f'/masks/pat{i}'

            if not os.path.exists(img_path): os.makedirs(img_path)
            if not os.path.exists(mask_path): os.makedirs(mask_path)
            x = (x + addition)*mult#/4095
            y = np.where(y>0,255,0)
            for n, (img_slice, mask_slice) in enumerate(zip(x,y)):
                #img_slice = (img_slice + 1024)*65535/4095
                mask_slice = mask_slice
                img = Image.fromarray(img_slice).convert("RGB")
                mask = Image.fromarray(mask_slice).convert("RGB")
                img.save(f'{img_path}/file{n}.png')
                mask.save(f'{mask_path}/file{n}.png')
    else:
        for i, (x) in enumerate(data):
            img_path = target_dir
            if not os.path.exists(target_dir): os.makedirs(target_dir)

            img = Image.fromarray(x).convert("RGB")
            img.save(f'{target_dir}/file{i}.png')       
                
class RegionGrow3D:
    
    """
    Funktion zur dreidimensionalen Regiongrowing.
    ---
    input:
        images: 3d Array, bei dem Regiongrowing betrieben werden soll
        masks:  3d Array, welches mit 0 und 1 die irrelevanten/relevanten Stellen des Bildes zeigt
                    irrelevant -> dort kein Regiongrowing
        upperThreshold: oberer Schwellenwert für die benachbarten Voxel
        lowerThreshold: unterer Schwellenwert für die benachbarten Voxel
        neighborMode: states wether to use just the direct neighbors ("6n") or the directional neighbors too ("26n")
        
    """
    
    def __init__(self, images, masks, upperThreshold,
                 lowerThreshold, neighborMode):
        self.images = images
        self.masks = masks
        self.outputMask = np.zeros_like(self.images)
        self.sz = images.shape[0]
        self.sy = images.shape[1]
        self.sx = images.shape[2]
        self.upperThreshold = upperThreshold
        self.lowerThreshold = lowerThreshold
        self.neighborMode = neighborMode
        self.queue = deque()
    
    def main(self, seed):
        newItem = seed
        
        self.outputMask[newItem[0], newItem[1], newItem[2]] = 1
        self.queue.append((seed[0], seed[1], seed[2]))
        
        while len(self.queue) != 0:
            newItem = self.queue.pop()
            if self.neighborMode == "26n":
                neighbors = [[newItem[0]-1, newItem[1]-1, newItem[2]-1],   [newItem[0]-1, newItem[1]-1, newItem[2]],   [newItem[0]-1, newItem[1]-1, newItem[2]+1],
                             [newItem[0]-1, newItem[1], newItem[2]-1],     [newItem[0]-1, newItem[1], newItem[2]],     [newItem[0]-1, newItem[1], newItem[2]+1],
                             [newItem[0]-1, newItem[1]+1, newItem[2]-1],   [newItem[0]-1, newItem[1]+1, newItem[2]],   [newItem[0]-1, newItem[1]+1, newItem[2]+1],
                             [newItem[0], newItem[1]-1, newItem[2]-1],     [newItem[0], newItem[1]-1, newItem[2]],     [newItem[0], newItem[1]-1, newItem[2]+1],
                             [newItem[0], newItem[1], newItem[2]-1],       [newItem[0], newItem[1], newItem[2]+1],     [newItem[0], newItem[1]+1, newItem[2]-1],
                             [newItem[0], newItem[1]+1, newItem[2]],       [newItem[0], newItem[1]+1, newItem[2]+1],   [newItem[0]+1, newItem[1]-1, newItem[2]-1],
                             [newItem[0]+1, newItem[1]-1, newItem[2]],     [newItem[0]+1, newItem[1]-1, newItem[2]+1], [newItem[0]+1, newItem[1], newItem[2]-1],
                             [newItem[0]+1, newItem[1], newItem[2]],       [newItem[0]+1, newItem[1], newItem[2]+1],   [newItem[0]+1, newItem[1]+1, newItem[2]-1],
                             [newItem[0]+1, newItem[1]+1, newItem[2]],     [newItem[0]+1, newItem[1]+1, newItem[2]+1]] 
                for neighbor in neighbors:
                    self.checkNeighbour(neighbor[0], neighbor[1], neighbor[2])
            elif self.neighborMode == "6n":
                self.checkNeighbour(newItem[0], newItem[1], newItem[2]-1)
                self.checkNeighbour(newItem[0], newItem[1], newItem[2]+1)
                self.checkNeighbour(newItem[0], newItem[1]-1, newItem[2])
                self.checkNeighbour(newItem[0], newItem[1]+1, newItem[2])
                self.checkNeighbour(newItem[0]-1, newItem[1], newItem[2])
                self.checkNeighbour(newItem[0]+1, newItem[1], newItem[2])
        return self.outputMask
        
    def checkNeighbour(self, z, y, x):
        if (x < self.sx and y < self.sy and z < self.sz 
            and x > -1 and y > -1 and z > -1 
            and self.masks[z,y,x] == 1):
            intensity = self.images[z, y, x]
            if self.isIntensityAcceptable(intensity) and self.outputMask[z,y,x] == 0:
                self.outputMask[z,y,x] = 1
                self.queue.append((z, y, x))
    
    def isIntensityAcceptable(self, intensity):
        if intensity < self.upperThreshold and intensity > self.lowerThreshold:
            return True
        return False  