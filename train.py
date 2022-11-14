from pickletools import optimize
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
import argparse

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
import torchvision.transforms as TF

import time
import copy

from collections import defaultdict

import wandb

Device = Any
LRScheduler = Any

import medicus

from medicus.utils import (
    timestamp, 
    to_device
)

from inference import inference
from evaluation import evaluate


def train(
    model: nn.Module,
    loss_fn: Callable,
    optimizer: optim.Optimizer,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    writer: medicus.logging.Writer,
    inference_samples: Optional[torch.Tensor] = None,
    inference_targets: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    num_epochs: int = 20,
    resume_epoch: int = 0,
    resume_step: int = 0,
    global_step: int = 0,
    log_every: int = 50,
    eval_every: int = 4_000,
    save_every: int = 20_000,
    save_dir: str = "runs/{}/checkpoints",
    device: Optional[str] = None,
) -> None:
    if None in (inference_samples, inference_targets):
        inference_samples, inference_targets = next(iter(test_dataloader))

    writer.images(
        "Images/inference/samples",
        inference_samples,
        global_step
    )
    writer.images(
        "Images/inference/targets",
        medicus.utils.masks2imgs(inference_targets),
        global_step
    )

    num_epoch_batches = len(train_dataloader)
    num_epoch_samples = len(train_dataloader.dataset)

    for i in range(resume_epoch, num_epochs):
        current_epoch = global_step * batch_size // num_epoch_samples
        desc = f"Training...[{current_epoch}/{num_epochs}]"
        tqdm_config = {"position": 0, "leave": False}
        with tqdm(train_dataloader, desc=desc, unit=" batch", **tqdm_config) as iterator:
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

                x = to_device(x, device)
                y = to_device(y, device)

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
                        model, inference_samples, device=device)

                    writer.images(
                        "Images/inference/outputs", 
                        medicus.utils.masks2imgs(outputs),
                        global_step
                    )

                    eval_loss = evaluate(
                        model, 
                        test_dataloader, 
                        loss_fn, 
                        device=device,
                        tqdm_config=tqdm_config
                    )
                    
                    iterator.set_postfix(eval_loss=eval_loss.item())
                    writer.scalar("Loss/eval", eval_loss, global_step)
                    
                if global_step % save_every == 0:
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "loss": loss,
                        "epoch": current_epoch,
                        "step": j,
                        "inference_samples": inference_samples,
                        "inference_samples": inference_targets
                    }, f"{save_dir}/ckpt_{global_step}")

                # Break the loop for infinite data loaders
                # if an an epoch is approximately completed
                if j >= num_epoch_batches - 1:
                    break


def main():
    # Parse the task specific arguments
    parser = medicus.utils.ArgumentParser(
        medicus.configs.default.arguments(),
    )
    args = parser.parse_args()

    # Load the configs for the respective components
    model_cfg = medicus.utils.load_cfg("model", args)
    optim_cfg = medicus.utils.load_cfg("optim", args)
    sched_cfg = medicus.utils.load_cfg("sched", args)
    data_cfg = medicus.utils.load_cfg("data", args)
    loss_cfg = medicus.utils.load_cfg("loss", args)

    # Prepare the directories for logging and saving
    run_id = timestamp()
    log_dir = args.log_dir.format(run_id)
    save_dir = args.save_dir.format(run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting new run with id: {run_id}")
    print(f"Saving logs at:           {log_dir}")
    print(f"Saving checkpoints at:    {save_dir}")

    config = medicus.utils.parse(
        args,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        sched_cfg=sched_cfg,
        data_cfg=data_cfg,
        loss_cfg=loss_cfg,
        log_dir=log_dir,
        save_dir=save_dir
    )
    print("Run with config:")
    print(json.dumps(config, indent=2))
    config_path = f"{log_dir}/config.json"
    print(f"Saving config at: {config_path}")
    with open(config_path, "w") as file:
        json.dump(config, file)

    # Initialize the writer for logging
    writer = medicus.logging.Writer(
        methods=args.logging_method,
        log_dir=log_dir,
        project=args.project,
    )

    # Prepare the data
    train_dataloader = medicus.data.load_data(data_cfg, "train")
    test_dataloader = medicus.data.load_data(data_cfg, "eval")

    # Retrieve the components implementations
    model = getattr(medicus.model, model_cfg["name"])
    optim = getattr(medicus.optim, optim_cfg["name"])
    sched = getattr(medicus.sched, sched_cfg["name"])
    loss = getattr(medicus.loss, loss_cfg["name"])

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO: let user specify device
    model = model(**model_cfg["config"])
    model = model.to(device)

    # Display the model
    sample, _ = next(iter(train_dataloader))
    summary(model, input_size=sample.shape[1:]) # TODO: where is the summary fn implemented?

    loss_fn = loss(**loss_cfg["config"])

    # Initialize optimizer and the learning rate schedule
    optimizer = optim(model.parameters(), **optim_cfg["config"])
    lr_scheduler = sched(optimizer, **sched_cfg["config"])

    inference_samples = None
    inference_targets = None
    resume_epoch = 0
    resume_step = 0
    global_step = 0

    # Resume the training components from a checkpoint
    if args.resume_from:
        state_dict = torch.load(args.resume_from)

        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        last_loss = state_dict["loss"]
        resume_epoch = state_dict["epoch"]
        resume_step = state_dict["step"]
        global_step = (resume_step + 1) + resume_epoch * len(train_dataloader)

        inference_samples = state_dict["inference_samples"]
        inference_targets = state_dict["inference_targets"]

        print(f"Resuming training from checkpoint at {args.resume_from}")
        print(f"    Last loss:             {last_loss}")
        print(f"    Resumed epoch:         {resume_epoch}")
        print(f"    Resumed step in epoch: {resume_step}")
        print(f"    Resumed global step:   {global_step}")

    train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        writer=writer,
        inference_samples=inference_samples,
        inference_targets=inference_targets,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        resume_epoch=resume_epoch,
        resume_step=resume_step,
        global_step=global_step,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        save_dir=save_dir,
        device=device
    )

if __name__ == "__main__":
    main()

