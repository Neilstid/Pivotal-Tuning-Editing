import numpy as np


def ridgelet_filter(scale, position, direction):
    i, j = np.indices((scale, scale))
    filter = np.power(scale, -1/2) * ridgelet((i * np.cos(direction) + j * np.sin(direction) - position) / scale)
    return filter / np.sum(filter)


def ridgelet(x):
    return np.exp(-(x ** 2) / 2) * -(1/2) * np.exp(-(x ** 2) / 8)