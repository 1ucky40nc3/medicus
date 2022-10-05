

import time
import torch
import torch.nn as nn

from typing import Tuple
from typing import List
from typing import Optional
from typing import Callable
from typing import Any
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm
from torch.utils import tensorboard
import wandb
import torchvision.transforms.functional as F
import numpy as np
from colour import Color


Device = Any

def timestamp() -> str:
    return time.strftime(
        "%Y%m%d%H%M%S", 
        time.localtime()
    )

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: Device,
    log_every: int = 50,
    desc: str = "Evaluating...",
    tqdm_config: dict = {}
) -> Any:
    model.eval()

    metric = MeanMetric()
    with torch.no_grad():
        with tqdm(dataloader, desc=desc, unit="batch", **tqdm_config) as iterator:
            for i, (x, y) in enumerate(iterator):
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                loss = loss_fn(outputs, y)
                metric.update(loss.cpu())

                if (i + 1) % log_every == 0:
                    iterator.set_postfix(
                        mean_loss=metric.compute().item())

    return metric.compute()

def inference(
    model: nn.Module,
    samples: torch.Tensor,
    device: Device
) -> Tuple[torch.Tensor]:
    with torch.no_grad():
        samples = samples.to(device)
        outputs = model(samples)

    return outputs

def parse(config) -> dict:
    return {
        "model": {
            "name": config["model"].__name__,
            "config": config["model_config"]
        },
        "loss_fn": config["loss_fn"].__name__,
        "optimizer": {
            "name": config["optimizer"].__name__,
            "config": config["optimizer_config"]
        },
        "lr_scheduler": {
            "name": config["lr_scheduler"].__name__,
            "config": config["lr_scheduler_config"]
        },
        "num_epochs": config["num_epochs"],
        "log_dir": config["log_dir"],
        "save_dir": config["save_dir"],
        "log_every": config["log_every"],
        "eval_every": config["eval_every"],
        "save_every": config["save_every"],
        "methods": config["methods"],
        "project": config["project"],
        "notes": config["notes"],
        "tags": config["tags"]
    }

class Logger:
    """A Logger class.

    Log scalars and images with Tensorboard or Weights & Biases.
    This makes the use of multiple methods very easy. Simply supply
    a list of the types of summary writer to log with.

    Attrs:
        types (list[str]): The types of summary writers to log to.
                           Possible values are ('tensorboard', 'w&b').
        log_dir (str): The directory to write log files to.
        project (optional, str): The project name to log to. (Needed for 'w&b')
        config (optional, str): A runs config. (Needed for 'w&b')
    """
    logger_methods: Tuple[str] = ("tensorboard", "w&b")

    def __init__(
        self,
        methods: Tuple[str],
        log_dir: str,
        project: Optional[str] = None,
        config: Optional[dict] = None,
        **kwargs
    ) -> None:
        assert all(t in self.logger_methods for t in methods), (
            "Error: A wrong item in `logger_methods` was supplied! " + 
            f"All possible logger types are: {self.logger_methods}")
        
        self.methods = set(methods)
        self.log_dir = log_dir
        self.project = project
        self.config = config

        for method in self.methods:
            attrs = {
                **kwargs,
                "log_dir": log_dir, 
                "project": project, 
                "config": config,
            }
            self.map("init", method, attrs)

    def init_tb(self, log_dir: str, **kwargs) -> None:
        self.tb_writer = tensorboard.SummaryWriter(log_dir=log_dir)

    def init_wb(self, project: str, config: str, **kwargs) -> None:
        assert project is not None, "Error: Weights & Biases logging shall be used, but no `project` is supplied!"
        assert config is not None, "Error: Weights & Biases logging shall be used, but no `config` is supplied!"

        wandb.init(project=project, config=config)

    def scalar_tb(self, name: str, value: float, step: int) -> None:
        self.tb_writer.add_scalar(name, value, step)

    def scalar_wb(self, name: str, value: float, step: int) -> None:
        wandb.log({name: value}, step=step)

    def images_tb(self, name: str, images: torch.Tensor, step: int) -> None:
        self.tb_writer.add_images(name, images, step)

    def images_wb(self, name: str, images: torch.Tensor, step: int) -> None:
        table = wandb.Table(columns=['ID', 'Image'])

        for id, img in zip(range(len(images)), images):
            img = F.to_pil_image(img)
            img = wandb.Image(img)
            table.add_data(id, img)

        wandb.log({name: table}, step=step)

    def map(self, action: str, method: str, attrs: dict) -> None:
        mapping = {
            "init": {
                "tensorboard": self.init_tb,
                "w&b": self.init_wb
            },
            "scalar": {
                "tensorboard": self.scalar_tb,
                "w&b": self.scalar_wb
            },
            "images": {
                "tensorboard": self.images_tb,
                "w&b": self.images_wb
            }
        }

        func = mapping[action][method]
        func(**attrs)

    def scalar(self, name: str, value: float, step: int) -> None:
        for method in self.methods:
            attrs = {
                "name": name, 
                "value": value, 
                "step": step
            }
            self.map("scalar", method, attrs)

    def images(self, name: str, images: torch.Tensor, step: int) -> None:
        for method in self.methods:
            attrs = {
                "name": name, 
                "images": images, 
                "step": step
            }
            self.map("images", method, attrs)

def masks_to_colorimg(masks):
    colors = np.asarray([
        (201, 58, 64), 
        (242, 207, 1), 
        (0, 152, 75), 
        (101, 172, 228), 
        (56, 34, 132), 
        (160, 194, 56)
    ])

    # shape: [H, W, 3]
    colorimg = np.ones(
        (masks.shape[1], masks.shape[2], 3), 
        dtype=np.float32
    ) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)
  

def masks2imgs(masks):
    masks = masks.cpu().numpy()
    batch, channels, height, width = masks.shape

    red, blue = Color("red"), Color("blue")
    colors = list(red.range_to(blue, channels))
    colors = np.array([c.rgb for c in colors]) * 255

    imgs = np.ones(
        (batch, height, width, 3), 
        dtype=np.float32
    ) * 255

    for i in range(batch):
        for y in range(height):
            for x in range(width):
                selected_colors = colors[masks[i, :, y, x] > 0.5]

                if len(selected_colors) > 0:
                    imgs[i, y, x, :] = np.mean(selected_colors, axis=0)

    imgs = imgs.transpose((0, 3, 1, 2))
    imgs = torch.from_numpy(imgs).contiguous()
    imgs = imgs.float().div(255)

    return imgs

