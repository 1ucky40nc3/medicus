from typing import Any
from typing import Dict
from typing import Tuple
from typing import Callable
from typing import Optional

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

import torchvision.transforms.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as TNF

import torchvision

import time
import copy

from collections import defaultdict

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

def train_model(
    model,
    optimizer,
    scheduler,
    dataloader,
    device,
    writer,
    num_epochs = 25,
    save_model = False,
    save_path = "",
    load_model = False,
    load_path = ""):


    
    """
    writer.add_scalar("Loss", total_loss, epoch)
    writer.add_scalar("Correct", total_correct, epoch)
    writer.add_scalar("Accuracy", total_correct/ len(train_set), epoch)
    writer.add_hparams(
            {"lr": lr, "bsize": batch_size, "shuffle":shuffle},
            {
                "accuracy": total_correct/ len(train_set),
                "loss": total_loss,
            },
        )

    TODO: Add train and test writer??"""

    if(load_model):
        model = torch.jit.load(load_path)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    writer.add_scalar("Loss/train", loss, epoch)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                writer.add_scalar("Loss/val", best_loss, epoch)
                writer.add_scalar("Dice/val", metrics['dice'], epoch)
                outputs_grid = torchvision.utils.make_grid(outputs)
                inputs_grid = torchvision.utils.make_grid(inputs)
                labels_grid = torchvision.utils.make_grid(labels)
                writer.add_image("prediction", outputs_grid)
                writer.add_image("images", inputs_grid)
                writer.add_image("truth", labels_grid)

                if(save_model):
                  model_scripted = torch.jit.script(model) # Export to TorchScript
                  model_scripted.save(save_path) 

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



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

    writer = tensorboard.SummaryWriter(log_dir=log_dir)

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

    writer.add_images(
        f"Images/eval_img_samples",
        img_tensor=test_samples,
        global_step=global_step)
    
    writer.add_images(
        f"Images/eval_img_targets",
        img_tensor=masks2imgs(test_targets),
        global_step=global_step)

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

                    iterator.set_postfix(
                        mean_loss=mean_loss.item())
                    writer.add_scalar(
                        "Loss/train_mean_log_step",
                        scalar_value=mean_loss, 
                        global_step=global_step)
                
                if global_step % eval_every == 0:
                    outputs = inference(
                        model, test_samples, device=device)

                    writer.add_images(
                        f"Images/eval_img_outputs",
                        img_tensor=masks2imgs(outputs),
                        global_step=global_step)

                    eval_mean_loss = evaluate(
                        model, 
                        eval_dataloader, 
                        loss_fn, 
                        device=device,
                        tqdm_config=tqdm_config)
                    
                    iterator.set_postfix(
                        eval_mean_loss=eval_mean_loss.item())
                    writer.add_scalar(
                        "Loss/eval_mean",
                        eval_mean_loss,
                        global_step=global_step)
                    
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