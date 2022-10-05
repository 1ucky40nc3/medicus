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
from colour import Color

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

from .objectives.unet import dice_loss
from .utils.utils_training import timestamp, parse, inference, evaluate, Logger, masks2imgs

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


