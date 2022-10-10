from typing import Tuple
from typing import Optional

import torch
from torch.utils import tensorboard

import torchvision.transforms.functional as F

import wandb


class Writer:
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