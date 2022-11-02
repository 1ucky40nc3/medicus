from typing import (
    Any,
    List,
    Dict,
    Tuple,
    Optional,
    Callable,
)

import os
import json
import time
import logging
import argparse

import flatten_dict

import numpy as np

import torch

from colour import Color



Device = Any
Module = Any


def timestamp() -> str:
    return time.strftime(
        "%Y%m%d%H%M%S", 
        time.localtime()
    )


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


class ArgumentParser:
    def __init__(self, *args) -> None:
        self.parser = argparse.ArgumentParser(parents=args)
    
    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()


def parse(
    args: argparse.Namespace, 
    **kwargs
) -> dict:
    config = {**vars(args), **kwargs}
    return config


def parse_config_arg(
    config: List[str],
) -> Dict[str, Any]:
    parsed = {}

    for entry in config:
        if os.path.isfile(entry):
            return json.load(open(entry))

        key, value = entry.split("=")
        path = tuple(key.split("."))
        parsed[path] = value

    return flatten_dict.unflatten(parsed)


def combine_dicts(
    a: Dict[str, Any],
    b: Dict[str, Any]
) -> Dict[str, Any]:
    a = flatten_dict.flatten(a)
    b = flatten_dict.flatten(b)

    c = {**a, **b}
    return flatten_dict.unflatten(c)


class FormatDict(dict):
    def __missing__(self, key):
        return "{%s}" % str(key)


def load_cfg(
    name: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
    cfg: Optional[dict] = None,
    config: Dict[Any, str] = {},
    **kwargs
) -> dict:
    if name is not None and args is not None:
        if args.config:
            config = parse_config_arg(args.config)
            config = config.get(name, {})

        cfg = getattr(args, name)
        cfg = json.load(open(cfg))

        cfg = combine_dicts(cfg, config)

    cfg = flatten_dict.flatten(cfg)

    for path, value in cfg.items():
        key = path[-1]
        if value is None and hasattr(args, key):
            cfg[path] = getattr(args, key)
        
        if isinstance(value, str):
            items = vars(args) if args is not None else {}
            items = FormatDict(**items, **kwargs)
            cfg[path] = value.format_map(items)

    cfg = flatten_dict.unflatten(cfg)
    print(json.dumps(cfg))
    return cfg


def get_cls(
    module: Module, 
    name: str
) -> Optional[Any]:
    try:
        return getattr(module, name)
    except AttributeError:
        logging.error(f"The class {name} couldn't be retrieved from {module}")
    return None