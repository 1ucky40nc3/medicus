from typing import List

import numpy as np
from perlin_noise import PerlinNoise


def generate_perlin_noise(
    height: int,
    width: int,
    octaves: List[int] = [10],
    scaling: List[float] = [1],
    seed: int = None,
) -> np.ndarray:
    assert len(octaves) == len(scaling)

    x = lambda i: i % width
    y = lambda i: i // height

    f = lambda i, noise: noise([x(i)/width, y(i)/height])
    vf = np.vectorize(f, excluded=["noise"])

    combined = np.zeros(height * width)
    for o, s in zip(octaves, scaling):
        noise = PerlinNoise(octaves=o, seed=seed)

        values = np.arange(height * width)
        values = vf(values, noise=noise)
        combined += values * s
    
    combined = combined.reshape((height, width, 1))
    combined = (combined - combined.min()) / (combined.max() - combined.min())

    return combined