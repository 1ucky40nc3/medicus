from typing import Any
from typing import Dict
from typing import Tuple
from typing import Callable
from typing import Optional
from typing import List

import os
import sys
import math
import json
import logging

from tqdm import tqdm

import numpy as np

import torch.nn as nn
from torch.utils import tensorboard
from torchmetrics import MeanMetric
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

import torchvision.transforms.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as TNF

import torchvision

import time
import copy

from collections import defaultdict

import wandb

Device = Any
LRScheduler = Any

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = TNF.binary_cross_entropy_with_logits(pred, target)
        
    pred = TNF.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def train(
    model: nn.Module,
    loss_fn: Callable,
    optimizer: optim.Optimizer,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    num_epochs: int = 20,
    model_config: Dict[str, Any] = {},
    optimizer_config: Dict[str, Any] = {},
    lr_scheduler_config: Dict[str, Any] = {},
    device: Optional[str] = None,
    log_dir: str = "runs/{}/logs",
    save_dir: str = "runs/{}/checkpoints",
    resume_from: Optional[str] = None,
    log_every: int = 50,
    eval_every: int = 4_000,
    save_every: int = 20_000,
    methods: Tuple[str] = ("tensorboard",),
    project: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> None:
    run_id = timestamp()
    log_dir = log_dir.format(run_id)
    save_dir = save_dir.format(run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting new run with id: {run_id}")
    print(f"Saving logs at:           {log_dir}")
    print(f"Saving checkpoints at:    {save_dir}")

    config = parse(locals())
    print("Run with config:")
    print(json.dumps(config, indent=2))
    config_path = f"{log_dir}/config.json"
    print(f"Saving config at: {config_path}")
    with open(config_path, "w") as file:
        json.dump(config, file)

    writer = Logger(**config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model(**model_config)
    model = model.to(device)

    sample, _ = next(iter(train_dataloader))
    summary(model, input_size=sample.shape[1:])

    optimizer = optimizer(model.parameters(), **optimizer_config)
    lr_scheduler = lr_scheduler(optimizer, **lr_scheduler_config)

    resume_epoch = 0
    resume_step = 0
    global_step = 0

    test_samples, test_targets = next(iter(eval_dataloader))

    if resume_from:
        state_dict = torch.load(resume_from)

        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        last_loss = state_dict["loss"]
        resume_epoch = state_dict["epoch"]
        resume_step = state_dict["step"]
        global_step = (resume_step + 1) + resume_epoch * len(train_dataloader)

        test_samples = state_dict["test_samples"]
        test_targets = state_dict["test_targets"]

        print(f"Resuming training from checkpoint at {resume_from}")
        print(f"    Last loss:             {last_loss}")
        print(f"    Resumed epoch:         {resume_epoch}")
        print(f"    Resumed step in epoch: {resume_step}")
        print(f"    Resumed global step:   {global_step}")

    writer.images("Images/test_samples", test_samples, global_step)
    writer.images("Images/test_targets", masks2imgs(test_targets), global_step)

    for i in range(resume_epoch, num_epochs):
        desc = f"Training...[{i + 1}/{num_epochs}]"
        tqdm_config = {"position": 0, "leave": False}
        with tqdm(train_dataloader, desc=desc, unit="batch", **tqdm_config) as iterator:
            metric = MeanMetric()

            for j, (x, y) in enumerate(iterator):
                # Skip forward until the resume step is reached.
                # If no checkpoint is provided this isn't invoked.
                if resume_step:
                    if resume_step == j:
                        resume_step = 0
                    else:
                        continue
                global_step = (j + 1) + i * len(iterator)

                model.train()
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                loss = loss_fn(outputs, y)

                if not math.isfinite(loss.item()) or torch.isnan(loss):
                    print(f"Loss is {loss.item()}... stopping training!")
                    return # Just return to stop training ¯\_(ツ)_/¯
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                metric.update(loss.cpu().item())

                if global_step % log_every == 0:
                    mean_loss = metric.compute()
                    metric = MeanMetric()

                    iterator.set_postfix(mean_loss=mean_loss.item())
                    writer.scalar("Loss/train", mean_loss, global_step)
                
                if global_step % eval_every == 0:
                    outputs = inference(
                        model, test_samples, device=device)

                    writer.images(
                        "Images/test_outputs", 
                        masks2imgs(outputs),
                        global_step)

                    eval_mean_loss = evaluate(
                        model, 
                        eval_dataloader, 
                        loss_fn, 
                        device=device,
                        tqdm_config=tqdm_config)
                    
                    iterator.set_postfix(
                        eval_mean_loss=eval_mean_loss.item())
                    writer.scalar(
                        "Loss/eval_mean",
                        eval_mean_loss,
                        global_step)
                    
                if global_step % save_every == 0:
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "loss": loss,
                        "epoch": i,
                        "step": j,
                        "test_samples": test_samples,
                        "test_targets": test_targets
                    }, f"{save_dir}/ckpt_{global_step}")

def evaluate_dice_loss(
    model,
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

                outputs = TNF.sigmoid(outputs)
                loss = loss_fn(outputs, y)
                #TODO: find out if necessary
                metric.update(loss.cpu())

                if (i + 1) % log_every == 0:
                    iterator.set_postfix(
                        mean_loss=metric.compute().item())

    return metric.compute()



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
