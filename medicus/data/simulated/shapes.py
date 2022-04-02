from typing import Any
from typing import Optional

import os

import numpy as np

from matplotlib import colors
from matplotlib import patches
import matplotlib.pyplot as plt

from PIL import Image

from medicus.data.simulated.noise import generate_perlin_noise


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


def randn(n: int, scale: float = 1.) -> np.ndarray:
    normal = np.random.normal(scale=scale, size=n)
    normal = vinside(normal, -1.5, 1.5)
    normal = (normal + 1.5) / 3

    return normal


def random_color(
    class_color: Optional[str] = None
) -> str:
    if class_color is None:
        return np.random.choice(COLORS)
    
    class_index = COLORS.index(class_color)
    p = np.ones(len(COLORS))
    p /= len(COLORS) - 1
    p[class_index] = 0

    return np.random.choice(COLORS, p=p)


def norm(v: np.ndarray) -> np.ndarray:
    return (v - v.min()) / (v.max() - v.min())


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


def square(
    x: float,
    y: float,
    d: float,
    b: float,
    c: float,
    a: float,
    **kwargs
) -> patches.Rectangle:
    return patches.Rectangle(
        xy=(x, y),
        width=d,
        height=d,
        angle=b,
        color=c,
        alpha=a
    )


def polygon(
    xy: np.ndarray,
    c: str,
    a: float,
    closed: bool = True,
    **kwargs
) -> patches.Polygon:
    return patches.Polygon(
        xy=xy,
        color=c,
        alpha=a,
        closed=closed,
    )


def circle(
    x: float,
    y: float,
    r: float,
    c: str,
    a: float,
    **kwargs
) -> patches.Circle:
    return patches.Circle(
        xy=(x, y),
        radius=r,
        color=c,
        alpha=a
    )


def wedge(
    x: float,
    y: float,
    r: float,
    t1: float,
    t2: float,
    w: float,
    c: str,
    a: float,
    **kwargs
) -> patches.Wedge:
    return patches.Wedge(
        center=(x, y),
        r=r,
        theta1=t1,
        theta2=t2,
        width=w,
        color=c,
        alpha=a
    )



def koch_snowflake(order, scale=10):
    """
    Return two lists x, y of point coordinates of the Koch snowflake.

    Parameters
    ----------
    order : int
        The recursion depth.
    scale : float
        The extent of the snowflake (edge length of the base triangle).

    source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill.html
    """
    def _koch_snowflake_complex(order):
        if order == 0:
            # initial triangle
            angles = np.array([0, 120, 240]) + 90
            return scale / np.sqrt(3) * np.exp(np.deg2rad(angles) * 1j)
        else:
            ZR = 0.5 - 0.5j * np.sqrt(3) / 3

            p1 = _koch_snowflake_complex(order - 1)  # start points
            p2 = np.roll(p1, shift=-1)  # end points
            dp = p2 - p1  # connection vectors

            new_points = np.empty(len(p1) * 4, dtype=np.complex128)
            new_points[::4] = p1
            new_points[1::4] = p1 + dp / 3
            new_points[2::4] = p1 + dp * ZR
            new_points[3::4] = p1 + dp / 3 * 2
            return new_points

    points = _koch_snowflake_complex(order)
    x, y = points.real, points.imag

    #x = norm(x)
    #y = norm(y)

    xy = np.zeros((len(x), 2))
    xy[:, 0] = x
    xy[:, 1] = y
    return xy


def distance(p: np.ndarray) -> float:
    return np.sqrt(p[0]**2 + p[1]**2)
vdistance = np.vectorize(distance, signature="(n)->()")


def softmax(
    value: np.ndarray,
    eps: float = 1.0
) -> np.ndarray:
    return np.exp(value * eps) / np.sum(np.exp(value))


def add_noise(
    v: float, 
    n: np.ndarray, 
) -> np.ndarray:
    i = int(v * len(n)) % len(n)
    return v + n[i, i]
vadd_noise = np.vectorize(add_noise, excluded=["n"])


def random_distort(
    xy: np.ndarray
) -> np.ndarray:
    distances = vdistance(xy)
    distances = softmax(distances, eps=np.random.rand())

    x = xy[:, 0]
    y = xy[:, 1]

    x *= distances
    y *= distances
    x = norm(x)
    y = norm(y)
   
    noise = generate_perlin_noise(
        height=100,
        width=100,
        octaves=[1, 10],
        scaling=[1., .05])
    noise = (noise - .5) * .2

    x = vadd_noise(x, n=noise)
    y = vadd_noise(y, n=noise)

    x = norm(x)
    y = norm(y)

    xy[:, 0] = x
    xy[:, 1] = y

    return xy


def snowflake(
    width: int,
    height: int
) -> np.ndarray:
    order = np.random.randint(2, 6)
    snowflake = koch_snowflake(order)
    snowflake = random_distort(snowflake)
    scale = np.random.rand()
    snowflake[:, 0] = snowflake[:, 0] * width
    snowflake[:, 1] = snowflake[:, 1] * height

    return snowflake


def random_gauss(
    num: int = 200,
    sigma: float = .5,
    mu: float = 0.
) -> np.ndarray:
    min_x, max_x = np.random.rand(2)
    min_y, max_y = np.random.rand(2)
    min_x *= -1
    min_y *= -1

    x = np.linspace(min_x, max_x, num)
    y = np.linspace(min_y, max_y, num)
    x, y = np.meshgrid(x, y)
    dist = np.sqrt(x**2 + y**2)

    return np.exp(-((dist - mu)**2 / ( 2.0 * sigma**2)))


def random_rect_args(
    width: int,
    height: int,
    min_rect_scale: float,
    max_rect_scale: float,
    class_color: Optional[str] = None,
    scale: float = 1.,
) -> dict:
    x, y = randn(2, scale)
    w, h = randn(2, scale)
    x = x * width
    y = y * height
    w = inside(w, min_rect_scale, max_rect_scale)
    h = inside(h, min_rect_scale, max_rect_scale)
    w = w * width
    h = h * height
    a = 45 * randn(1)

    c = random_color(class_color)

    return locals()


def random_square_args(
    width: int,
    height: int,
    min_square_scale: float = 0.1,
    max_square_scale: float = 0.4,
    class_color: Optional[str] = None,
    scale: float = 1.,
) -> dict:
    x, y = randn(2, scale)
    d = np.random.rand()
    d = inside(d, min_square_scale, max_square_scale)
    d *= min(width, height)

    b = 45 * np.random.rand()
    
    c = random_color(class_color)
    a = np.random.rand()
    a = inside(a, 0.2, 0.8)

    return locals()


def random_polygon_args(
    width: int,
    height: int,
    min_polygon_scale: float = 0.1,
    max_polygon_scale: float = 0.7,
    min_points: int = 3,
    max_points: int = 8,
    class_color: Optional[str] = None,
) -> dict:
    n = np.random.randint(min_points, max_points)
    xy = np.random.rand(n, 2)
    xy[:, 0] = xy[:, 0] * width
    xy[:, 1] = xy[:, 1] * height
    
    s = np.random.rand()
    s = inside(s, min_polygon_scale, max_polygon_scale)
    xy *= s

    c = random_color(class_color)
    a = np.random.rand()
    a = inside(a, 0.2, 0.8)

    return locals()


def random_circle_args(
    width: int,
    height: int,
    min_circle_scale: float = 0.1,
    max_circle_scale: float = 0.4,
    class_color: Optional[str] = None,
) -> dict:
    x, y = np.random.rand(2)
    x *= width
    y *= height

    r = np.random.rand()
    r = inside(r, min_circle_scale, max_circle_scale)
    r *= min(width, height)

    c = random_color(class_color)
    a = np.random.rand()
    a = inside(a, 0.2, 0.8)

    return locals()


def random_wedge_args(
    width: int,
    height: int,
    min_circle_scale: float = 0.1,
    max_circle_scale: float = 0.4,
    class_color: Optional[str] = None,
) -> dict:
    x, y = np.random.rand(2)
    x *= width
    y *= height

    r = np.random.rand()
    r = inside(r, min_circle_scale, max_circle_scale)
    r *= min(width, height)

    t1 = 360. * np.random.rand()
    t2 = 360. * np.random.rand()

    w = np.random.rand()
    w = r * w

    c = random_color(class_color)
    a = np.random.rand()
    a = inside(a, 0.2, 0.8)

    return locals()


def save_figure_axes(
    fig: Figure,
    axes: Axes, 
    file_path: str
) -> None:
    extend = axes.get_window_extent(
        ).transformed(fig.dpi_scale_trans.inverted())
    extend.x0 = 2.7364705882352944
    extend.x1 = 3.823529411764706
    fig.savefig(file_path, bbox_inches=extend)


def generate_rectangles_samples(
    height: int,
    width: int,
    num_samples: int = 20,
    num_rects: int = 20,
    class_color: str = "red",
    min_rect_scale: float = 0.1,
    max_rect_scale: float = 0.4,
    directory: str = ".",
    file_prefix: str = "rectangles",
    file_type: str = "png",
    seed: int = None
) -> None:
    assert class_color in COLORS

    if seed:
        np.random.seed(seed)

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
        for j in range(num_rects - 1):
            patch = rectangle(
                **random_rect_args(*args))
            inp.add_patch(patch)
        
        rect_args = random_rect_args(*args, scale=0.5)
        inp_rect_args = {**rect_args, **{"c": class_color}}
        tgt_rect_args = {**rect_args, **{"c": "white"}}

        inp.add_patch(rectangle(**inp_rect_args))
        tgt.add_patch(rectangle(**tgt_rect_args))
        
        inp_path = os.path.join(directory, "input", f"{file_prefix}_{i}.{file_type}")
        tgt_path = os.path.join(directory, "target", f"{file_prefix}_{i}.{file_type}")

        save_figure_axes(fig, inp, inp_path)
        save_figure_axes(fig, tgt, tgt_path)


# generate_rectangles_samples(200, 200, 10)
shapes_mapping = {
    "square": {
        "args": random_square_args,
        "func": square
    },
    "polygon": {
        "args": random_polygon_args,
        "func": polygon
    },
    "circle": {
        "args": random_circle_args,
        "func": circle
    },
    "wedge": {
        "args": random_wedge_args,
        "func": wedge
    }
}


def generate_shapes_samples(
    height: int,
    width: int,
    num_samples: int,
    num_shapes: int = 20,
    class_shape: str = "square",
    directory: str = ".",
    file_prefix: str = "shapes",
    file_type: str = "png",
    seed: int = None
) -> None:
    if seed:
        np.random.seed(seed)

    inp_background = np.ones((height, width, 3))
    tgt_background = np.zeros((height, width, 1))

    os.makedirs(os.path.join(directory, "input"), exist_ok=True)
    os.makedirs(os.path.join(directory, "target"), exist_ok=True)

    other_shapes = list(shapes_mapping.keys())
    other_shapes.remove(class_shape)

    for i in range(num_samples):
        fig, (_, inp, tgt) = plt.subplots(3)
        _.imshow(inp_background)

        inp.imshow(inp_background)
        tgt.imshow(tgt_background)

        for j in range(num_shapes - 1):
            shape = np.random.choice(other_shapes)
            args = shapes_mapping[shape]["args"]
            func = shapes_mapping[shape]["func"]

            args = args(width, height)
            shape = func(**args)

            inp.add_patch(shape)
        
        args = shapes_mapping[class_shape]["args"]
        func = shapes_mapping[class_shape]["func"]
        args = args(width, height)
        
        inp_args = {**args, "a": inside(args["a"], .6, 1.)}
        tgt_args = {**args, "c": "white", "a": None}

        inp.add_patch(func(**inp_args))
        tgt.add_patch(func(**tgt_args))

        inp_path = os.path.join(directory, "input", f"{file_prefix}_{i}.{file_type}")
        tgt_path = os.path.join(directory, "target", f"{file_prefix}_{i}.{file_type}")

        save_figure_axes(fig, inp, inp_path)
        save_figure_axes(fig, tgt, tgt_path)


#generate_shapes_samples(200, 200, 10)


def generate_snowflakes_samples(
    height: int,
    width: int,
    num_samples: int = 20,
    num_snowflakes: int = 5,
    class_color: str = "red",
    directory: str = ".",
    file_prefix: str = "snowflakes",
    file_type: str = "png",
    seed: Optional[int] = None
) -> None:
    if seed:
        np.random.seed(seed)

    inp_background = np.ones((height, width, 3))
    tgt_background = np.zeros((height, width, 1))

    os.makedirs(os.path.join(directory, "input"), exist_ok=True)
    os.makedirs(os.path.join(directory, "target"), exist_ok=True)

    for i in range(num_samples):
        fig, (inp, tgt) = plt.subplots(2)

        inp.imshow(inp_background)
        tgt.imshow(tgt_background)

        for j in range(num_snowflakes - 1):
            shape = snowflake(width, height)
            color = random_color(class_color)
            patch = patches.Polygon(
                xy=shape, 
                closed=True,
                color=color)
            inp.add_patch(patch)
        
        shape = snowflake(width, height)
        args = dict(xy=shape, closed=True)
        
        inp_patch = patches.Polygon(**args, color=class_color)
        tgt_patch = patches.Polygon(**args, color="white")

        inp.add_patch(inp_patch)
        tgt.add_patch(tgt_patch)

        inp_path = os.path.join(directory, "input", f"{file_prefix}_{i}.{file_type}")
        tgt_path = os.path.join(directory, "target", f"{file_prefix}_{i}.{file_type}")

        save_figure_axes(fig, inp, inp_path)
        save_figure_axes(fig, tgt, tgt_path)


#generate_snowflakes_samples(200, 200, 10)


def generate_gauss_samples(
    dim: int,
    num_samples: int = 20,
    directory: str = ".",
    file_prefix: str = "gauss",
    file_type: str = "png",
    seed: Optional[int] = None
) -> None:
    if seed:
        np.random.seed(seed)

    os.makedirs(os.path.join(directory, "input"), exist_ok=True)
    os.makedirs(os.path.join(directory, "target"), exist_ok=True)

    for i in range(num_samples):
        gauss = random_gauss(dim)
        noise = generate_perlin_noise(
            height=dim, 
            width=dim,
            octaves=[1., 4., 10.], 
            scaling=[1., 1, .125])
        noise = noise.reshape((dim, dim))
        noise = norm(noise) * 0.7

        gauss = softmax(gauss)
        gauss = norm(gauss)

        inp_img = gauss * .7 + noise * .3
        inp_img *= 255
        tgt_img = gauss > .92

        inp_path = os.path.join(directory, "input", f"{file_prefix}_{i}.{file_type}")
        tgt_path = os.path.join(directory, "target", f"{file_prefix}_{i}.{file_type}")

        Image.fromarray(inp_img).convert("RGB").save(inp_path)
        Image.fromarray(tgt_img).convert("RGB").save(tgt_path)


generate_gauss_samples(108, 10)