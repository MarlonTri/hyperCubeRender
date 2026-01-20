import colorsys
import numpy as np


def equicolor(n, alpha=False):
    cols = [colorsys.hsv_to_rgb((i + 0.5) / n, 0.67, 1) for i in range(n)]
    return np.array(cols)


def equicolor_with_white(n, alpha=False):
    return np.vstack([equicolor(n - 1), [(1, 1, 1)]])


def color_aesthetic(n, alpha=False):
    colors = equicolor(n + 1)


class ColorProjector(object):
    def __init__(self, n_dim, alpha=False):
        if alpha:
            raise Exception
        self.colors = equicolor_with_white(n_dim + 1)

    def __call__(self, *args):
        return self.transform(*args)

    def transform(self, arr):
        X_norm = arr + 0.001
        weights = X_norm / X_norm.sum(axis=1, keepdims=True)  # shape (n_points, n_dims)
        colors = weights @ self.colors[: X_norm.shape[1]]  # shape (n_points, 4)
        colors = np.clip(colors, a_min=0, a_max=1)  ## eliminate fp errors
        return colors
