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
    model,
    loss_fn: Callable,
    optimizer: optim.Optimizer,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    add_dice: bool = False,
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
    #model = model(**model_config)
    model = model.to(device)

    sample, _ = next(iter(train_dataloader))
    summary(model, input_size=sample.shape[1:])

    #optimizer = optimizer(model.parameters(), **optimizer_config)
    #lr_scheduler = lr_scheduler(optimizer, **lr_scheduler_config)
    #TODO: a lot

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
        img_tensor=test_targets,
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
                        mean_loss = mean_loss.item())
                    writer.add_scalar(
                        "Loss/train_mean_log_step",
                        scalar_value = mean_loss, 
                        global_step = global_step)
                
                if global_step % eval_every == 0:
                    outputs = inference(
                        model, test_samples, device=device)

                    writer.add_images(
                        f"Images/eval_img_outputs",
                        img_tensor=outputs,
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

                    if(add_dice):
                        eval_mean_dice = evaluate_dice_loss(
                        model, 
                        eval_dataloader, 
                        dice_loss, 
                        device=device,
                        tqdm_config=tqdm_config)
                            
                        writer.add_scalar(
                            "Dice/val",
                            eval_mean_dice,
                            global_step = global_step)
                    
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
                loss = loss_fn(outputs, y)
                metric.update(loss.cpu())

                if (i + 1) % log_every == 0:
                    iterator.set_postfix(
                        mean_loss=metric.compute().item())

    return metric.compute()

def inference(
    model,
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
            "name": "Test",#config["model"].__name__
            "config": config["model_config"]
        },
        "loss_fn": config["loss_fn"].__name__,
        "optimizer": {
            "name": "Test", #config["optimizer"].__name__,
            "config": config["optimizer_config"]
        },
        "lr_scheduler": {
            "name": "Test",#config["lr_scheduler"].__name__,
            "config": config["lr_scheduler_config"]
        },
        "num_epochs": config["num_epochs"],
        "log_dir": config["log_dir"],
        "save_dir": config["save_dir"],
        "log_every": config["log_every"],
        "eval_every": config["eval_every"],
        "save_every": config["save_every"]
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
  

def masks2imgs(masks: torch.Tensor) -> torch.Tensor:
    masks = masks.cpu().numpy()

    imgs = []
    for mask in masks:
        img = masks_to_colorimg(mask)
        img = F.to_tensor(img)
        imgs.append(img)
    
    return torch.stack(imgs)