import time
from matplotlib import pyplot as plt
import numpy as np
from colorProjector import ColorProjector
from spaceProjector import ManifoldSpaceProjector


class Renderer(object):
    def __init__(
        self,
        space_projector: ManifoldSpaceProjector,
        color_projector: ColorProjector,
        space_sampler,
        resolution,
    ):
        self.space_projector = space_projector
        self.color_projector = color_projector
        self.space_sampler = space_sampler
        self.resolution = resolution

        self.img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        self.img_sum = np.zeros((resolution, resolution, 3))
        self.img_weights = np.zeros((resolution, resolution, 1), dtype=np.uint8)

    def init_plt(self):
        plt.ion()
        fig, ax = plt.subplots()
        self.plt_im = ax.imshow(self.img_sum)

    def update_plt(self):
        self.plt_im.set_data(self.img)
        plt.pause(1)

    def render_step(self, n_batches):
        tc = time.time()
        for i in range(n_batches):
            X = next(self.space_sampler)

            coords = (self.resolution * self.space_projector.transform(X)).astype(int)
            coords = np.clip(coords, a_min=0, a_max=self.resolution - 1)
            xs = coords[:, 0]
            ys = coords[:, 1]

            self.img_sum[xs, ys] = self.img_sum[xs, ys] + self.color_projector(X)
            self.img_weights[xs, ys] = self.img_weights[xs, ys] + 1

        tc = time.time() - tc
        print(f"{i=}, {tc=}")
        self.img = self.img_sum / (
            np.reshape(
                np.max(self.img_sum, axis=2), (self.resolution, self.resolution, 1)
            )
        )
        self.img[np.all(self.img_sum < 0.02, axis=2)] = 0

        return self.img
