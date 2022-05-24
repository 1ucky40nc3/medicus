import os
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Optional

import torch.nn as nn

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


def predict_pat(
    model: nn.Module,
    model_config: Dict[str, Any] = {},
    device: Optional[str] = None,
    dataloader: DataLoader = Callable,
    save_dir: str = "runs/{}/checkpoints",
    resume_from: str = "runs/{}/checkpoints/{}",
) -> None:

    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading model from: {resume_from}")
    print(f"Saving predictions at: {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model(**model_config)
    model = model.to(device)

    state_dict = torch.load(resume_from)

    model.load_state_dict(state_dict["model"])

    model.eval()
   
    n = 0
    i = 0

    for x,y in dataloader:
        x = x.to(device)
        outputs = model(x)


        outputs = outputs.cpu().detach()
        for pred in outputs:
            pred =  (pred[0] > 0.5).float().numpy() * 255
            img = Image.fromarray(pred).convert("RGB")
            img.save(f'{save_dir}/file{n}.png')
            n = n + 1

        for pred in y:
            pred =  (pred[0] > 0.5).float().numpy() * 255
            img = Image.fromarray(pred).convert("RGB")
            img.save(f'C:/Users/Max Beyer/ML_data/predictions2/mask{i}.png')
            i = i + 1

