from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Iterable
from typing import Optional

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image


def predict_pat(
    model: nn.Module,
    dataloader: DataLoader,
    save_dir: str = "./",
    resume_from: str = "./",
    model_config: Dict[str, Any] = {},
    device: Optional[str] = None,
) -> None:
    """Make a prediction with a selected model.
    
    Args:
        model (nn.Module): The pretrained model.
        model_config (dict): The model config.
        device (str): The device to run with.
        dataloader ()
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading model from: {resume_from}")
    print(f"Saving predictions at: {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model(**model_config)

    state_dict = torch.load(resume_from)
    model.load_state_dict(state_dict["model"])
    
    model = model.to(device)

    model.eval()
   
    n = 0
    i = 0

    for x, y in dataloader:
        x = x.to(device)
        outputs = model(x)


        outputs = outputs.cpu().detach()
        for pred in outputs:
            pred =  (pred[0] > 0.5).float().numpy() * 255
            img = Image.fromarray(pred).convert("RGB")
            img.save(f'{save_dir}/file{n}.png')
            n = n + 1


