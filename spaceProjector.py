import numpy as np

import sklearn.manifold as manifold
import spaceSampler
from scipy.interpolate import RegularGridInterpolator

LLE_METHODS = []


class SpaceProjector(object):
    def __init__(self): ...


class InterpolatedSpaceProjector(SpaceProjector):
    def __init__(self, resolution, input_dim, space_projector):
        self.grid = spaceSampler.unit_grid(
            resolution, input_dim, hollow=False, flattened=True
        )

        X_projected = space_projector(self.grid)
        X_projected_reshaped = X_projected.reshape([resolution] * input_dim + [2])

        self.pos_interpolator = RegularGridInterpolator(
            [np.mgrid[0: resolution] / (resolution - 1) for _ in range(input_dim)],
            X_projected_reshaped,
        )

    def __call__(self, *args):
        return self.transform(*args)

    def transform(self, X):
        return self.pos_interpolator(X)


class ManifoldSpaceProjector(SpaceProjector):
    def __init__(self, method, input_dim, X):

        self.method = method.lower()
        self.input_dim = input_dim
        self.n_neighbors = 2 * input_dim - 1

        self.fitter = self._fit(X)
        X_p = self.fitter.transform(X)
        self.bounds_min = X_p.min(axis=0)
        self.bounds_max = X_p.max(axis=0)

    def _fit(self, X):

        if self.method in LLE_METHODS:
            return manifold.LocallyLinearEmbedding(
                n_neighbors=self.n_neighbors,
                n_components=2,
                method=self.method,
                random_state=42,
            ).fit(X)
        elif self.method == "isomap":
            isomap = manifold.Isomap(
                n_neighbors=self.n_neighbors, n_components=2, p=1, eigen_solver="dense"
            )
            return isomap.fit(X)

    def __call__(self, *args):
        return self.transform(*args)

    def transform(self, X):
        X_p = self.fitter.transform(X)
        X_p = X_p - self.bounds_min
        X_p = X_p / self.bounds_max / 2  # TODO - wtf?
        return X_p
