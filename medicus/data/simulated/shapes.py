from typing import Any

import os

import numpy as np

from matplotlib import colors
from matplotlib import patches
import matplotlib.pyplot as plt

Figure = Any
Axes = Any

COLORS = list(colors.CSS4_COLORS.keys())


def inside(
    val: float, 
    lower_lim: float, 
    upper_lim: float
) -> float:
    val = min(val, upper_lim)
    val = max(val, lower_lim)
    return val

vinside = np.vectorize(
    inside, excluded=["lower_lim", "upper_lim"])


def randn(n: int) -> np.ndarray:
    normal = np.random.normal(size=n)
    normal = vinside(normal, -1.5, 1.5)
    normal = (normal + 1.5) / 3

    return normal


def random_color(
    class_color: str
) -> str:
    class_index = COLORS.index(class_color)
    p = np.ones(len(COLORS))
    p /= len(COLORS) - 1
    p[class_index] = 0

    return np.random.choice(COLORS, p=p)


def rectangle(
    x: float,
    y: float,
    w: float,
    h: float,
    a: float,
    c: float,
    **kwargs
) -> patches.Rectangle:
    return patches.Rectangle(
        xy=(x, y),
        width=w,
        height=h,
        angle=a,
        color=c
    )


def random_rect_args(
    width: int,
    height: int,
    min_rect_scale: float,
    max_rect_scale: float,
    class_color: str
) -> dict:
    x, y = randn(2)
    w, h = randn(2)
    x = x * width
    y = y * height
    w = inside(w, min_rect_scale, max_rect_scale)
    h = inside(h, min_rect_scale, max_rect_scale)
    w = w * width
    h = h * height
    a = 45 * randn(1)
    c = random_color(class_color)

    return locals()


def save_figure_axes(
    fig: Figure,
    axes: Axes, 
    file_path: str
) -> None:
    extend = axes.get_window_extent(
        ).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(file_path, bbox_inches=extend)


def generate_rectangles_samples(
    height: int,
    width: int,
    num_samples: int,
    num_squares: int = 20,
    class_color: str = "red",
    min_rect_scale: float = 0.1,
    max_rect_scale: float = 0.4,
    directory: str = ".",
    file_prefix: str = "squares",
    file_type: str = "png"
) -> None:
    assert class_color in COLORS

    inp_background = np.ones((height, width, 3))
    tgt_background = np.zeros((height, width, 1))

    args = (
        width, 
        height, 
        min_rect_scale, 
        max_rect_scale, 
        class_color
    )

    os.makedirs(os.path.join(directory, "input"), exist_ok=True)
    os.makedirs(os.path.join(directory, "target"), exist_ok=True)

    for i in range(num_samples):
        fig, (inp, tgt) = plt.subplots(2)

        inp.imshow(inp_background)
        tgt.imshow(tgt_background)
        for j in range(num_squares - 1):
            patch = rectangle(
                **random_rect_args(*args))
            inp.add_patch(patch)
        
        rect_args = random_rect_args(*args)
        inp_rect_args = {**rect_args, **{"c": class_color}}
        tgt_rect_args = {**rect_args, **{"c": "white"}}

        inp.add_patch(rectangle(**inp_rect_args))
        tgt.add_patch(rectangle(**tgt_rect_args))
        
        inp_path = os.path.join(directory, "input", f"{file_prefix}_{i}.{file_type}")
        tgt_path = os.path.join(directory, "target", f"{file_prefix}_{i}.{file_type}")

        save_figure_axes(fig, inp, inp_path)
        save_figure_axes(fig, tgt, tgt_path)


generate_rectangles_samples(200, 200, 10)