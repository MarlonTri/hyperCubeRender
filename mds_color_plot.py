import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold as manifold
from scipy.interpolate import RegularGridInterpolator
from PIL import Image, ImageColor
import random
import time
import colorsys


# Variables for manifold learning.
N_NEIGHBORS = 3**4 - 1
N_SAMPLES = 1000
LLE_METHODS = []
COLORS = np.array(
    [
        [1, 0, 0, 1],  # Red
        [0, 1, 0, 1],  # Green
        [0, 0, 1, 1],  # Blue
        [1, 1, 0, 1],  # Yellow
        [0, 1, 1, 1],  # Orange
    ],
    dtype=float,
)  # shape (n_colors, 4)

COLORS = np.array([colorsys.hsv_to_rgb(i / 6, 1, 1) + (0,) for i in range(6)])


def manifold_learn_raw(X, method, input_dim):
    n_neighbors = 2 * input_dim - 1

    method = method.lower()
    if method in LLE_METHODS:
        return manifold.LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=2, method=method, random_state=42
        ).fit_transform(X)
    elif method == "isomap":
        isomap = manifold.Isomap(
            n_neighbors=n_neighbors, n_components=2, p=1  # , eigen_solver="dense"
        )
        return isomap.fit_transform(X)


def manifold_learn(X, method, input_dim):
    x = manifold_learn_raw(X, method, input_dim)
    x = x - x.min(axis=0)
    x = x / x.max(axis=0)
    return x


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


def gen_colors(arr):
    X_norm = (arr - arr.min(axis=0)) / (
        arr.max(axis=0) - arr.min(axis=0)
    )  # shape (n_points, n_dims)
    extra_col = (X_norm.shape[1] - X_norm.sum(axis=1, keepdims=True)) / X_norm.shape[1]
    X_norm = np.concatenate([X_norm, extra_col], axis=1)
    X_norm = X_norm / X_norm.max(axis=0)
    weights = X_norm / X_norm.sum(axis=1, keepdims=True)  # shape (n_points, n_dims)
    colors = weights @ COLORS[: X_norm.shape[1]]  # shape (n_points, 4)
    colors = np.clip(colors, a_min=0, a_max=1)  ## eliminate fp errors
    return colors


RADIUS = 2
N_DIMENSIONS = 5


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


def render_image(rad, n_dim, rez=2000):
    X = centered_mgrid(rad, n_dim, hollow=False)

    X_ml = manifold_learn(X, "isomap", n_dim)

    X_ml_reshaped = X_ml.reshape([rad * 2 + 1] * n_dim + [2])
    pos_interpolator = RegularGridInterpolator(
        [np.mgrid[-rad : rad + 1] for _ in range(n_dim)], X_ml_reshaped
    )

    colors = gen_colors(X)
    colors_reshaped = colors.reshape([rad * 2 + 1] * n_dim + [4])
    color_interpolator = RegularGridInterpolator(
        [np.mgrid[-rad : rad + 1] for _ in range(n_dim)], colors_reshaped
    )

    img_sum = np.zeros((rez, rez, 3), dtype=np.uint8) + 0.01

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(img_sum)
    tc = time.time()
    for i, pts in enumerate(grid_iterator(rad, n_dim, 10000)):

        coords = (rez * pos_interpolator(pts)).astype(int)
        coords = np.clip(coords, a_min=0, a_max=rez - 1)
        xs = coords[:, 0]
        ys = coords[:, 1]

        img_sum[xs, ys] = img_sum[xs, ys] + color_interpolator(pts)[:, :3]

        if i % 100 == 0:
            tc = time.time() - tc
            print(f"{i=}, {tc=}")
            img = img_sum / (
                np.reshape(np.max(img_sum, axis=2), (rez, rez, 1)) + i // 4e2
            )
            img[np.all(img_sum < 0.02, axis=2)] = 0
            im.set_data(img)
            plt.pause(1)
            tc = time.time()

    return X_ml, colors


render_image(RADIUS, N_DIMENSIONS, 2000)


# X = centered_mgrid(RADIUS, N_DIMENSIONS, hollow=True)
# X_ml = manifold_learn(X, "isomap", N_DIMENSIONS)
# colors = gen_colors(X_ml)

# # Plot
# plt.figure(figsize=(6, 6))
# plt.scatter(X_ml[:, 0], X_ml[:, 1], c=colors, s=5)
# plt.title("MDS embedding of 4D grid with color per dimension")
# plt.axis("equal")
# plt.show()

print()
