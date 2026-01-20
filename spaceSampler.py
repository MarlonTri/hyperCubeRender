import random

import numpy as np

def combo_iterator(iterators):
    while True:
        for iterator in iterators:
            yield next(iterator)

def grid_iterator(rad, n_dim, batch, hollow=True):
    while True:
        # onlyy apply rad at end, do multiply of [1, -1, 1, ...] type array so 1/n root can be applied
        x = np.random.random(n_dim * batch) * 2 - 1
        x = np.reshape(x, (batch, n_dim)) * rad
        if hollow:
            next = 0
            while next < 0.8:
                x[:, random.randint(0, n_dim - 1)] = (
                    random.randint(0, 1) * 2 - 1
                ) * rad
                next = random.random()

        yield x


def unit_grid_iterator(resolution, n_dim, batch, sample_dim=2, edge_bias=10):
    while True:
        x = np.random.random(n_dim * batch)
        x = np.reshape(x, (batch, n_dim))
        if sample_dim == n_dim or None:
            yield x
        else:
            n_clamped_dim = n_dim - sample_dim
            if edge_bias:
                clamped_dims = np.random.random(batch * n_clamped_dim)
                clamped_dims = clamped_dims.reshape(batch, n_clamped_dim)
                clamped_dims = np.pow(clamped_dims, edge_bias)
                edge_picks = np.random.randint(0, 2, n_clamped_dim)
                clamped_dims = (1 - clamped_dims) * edge_picks + clamped_dims * (
                    1 - edge_picks
                )

            else:
                clamped_dims = np.random.randint(0, 2, n_clamped_dim)

            x[:, :n_clamped_dim] = clamped_dims
            x = x[:, np.random.permutation(n_dim)]
            yield x


def unit_grid(resolution, dimension, hollow=True, flattened=True):

    if not flattened and hollow:
        raise Exception()

    slices = [slice(0, resolution)] * dimension
    grid = np.mgrid[*slices] / (resolution - 1)
    if not flattened:
        return grid
    lst = np.reshape(grid.T, (resolution**dimension, dimension))
    if not hollow:
        return lst
    return np.array([x for x in lst if np.min(x) == 0 or np.max(x) == 1])


def centered_mgrid(radius, dimension, hollow=True, flattened=True):

    if not flattened and hollow:
        raise Exception()

    slices = [slice(-radius, radius + 1)] * dimension
    grid = np.mgrid[*slices]
    if not flattened:
        return grid
    lst = np.reshape(grid.T, ((radius * 2 + 1) ** dimension, dimension))
    if not hollow:
        return lst
    return np.array([x for x in lst if np.max(np.abs(x)) == radius])
