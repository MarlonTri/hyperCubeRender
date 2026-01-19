import colorsys
import numpy as np


def equicolor(n, alpha=False):
    return np.array([colorsys.hsv_to_rgb(i / n, 1, 1) for i in range(n)])


class ColorProjector(object):
    def __init__(self, n_dim, alpha=False):
        if alpha:
            raise Exception
        self.colors = equicolor(n_dim + 1)

    def __call__(self, *args):
        return self.transform(*args)

    def transform(self, arr):
        X_norm = arr
        extra_col = (
            X_norm.shape[1] - X_norm.sum(axis=1, keepdims=True)
        ) / X_norm.shape[1]
        X_norm = np.concatenate([X_norm, extra_col], axis=1)
        X_norm[:, -1] = X_norm[:, -1] / X_norm[:, -1].max()
        weights = X_norm / X_norm.sum(axis=1, keepdims=True)  # shape (n_points, n_dims)
        colors = weights @ self.colors[: X_norm.shape[1]]  # shape (n_points, 4)
        colors = np.clip(colors, a_min=0, a_max=1)  ## eliminate fp errors
        return colors
