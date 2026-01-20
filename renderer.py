import time
from matplotlib import pyplot as plt
import numpy as np
from colorProjector import ColorProjector
from spaceProjector import ManifoldSpaceProjector

DEBUG = False

class ImageCompiler(object):
    def __init__(self, tot_sample_damp=0.0, peak_bright_percentile=80):
        self.tot_sample_damp = tot_sample_damp
        self.peak_bright_percentile = peak_bright_percentile

    def __call__(self, img_sum, img_weights):
        resolution = img_sum.shape[0]
        img = np.zeros((resolution, resolution, 3))

        top_bright = np.percentile(
            img_weights[img_weights != 0], q=self.peak_bright_percentile
        )
        sample_damp = np.clip(img_weights / top_bright, a_min=0, a_max=1)

        img = img_sum / img_weights
        img = np.nan_to_num(img, nan=0)
        img = img * sample_damp

        return img


IMAGE_COMPILER = ImageCompiler()


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
        temp_img_sum = np.zeros((self.resolution, self.resolution, 3))
        temp_img_weights = np.zeros(
            (self.resolution, self.resolution, 1), dtype=np.uint8
        )
        for i in range(n_batches):
            X = next(self.space_sampler)

            coords = (self.resolution * self.space_projector.transform(X)).astype(int)
            coords = np.clip(coords, a_min=0, a_max=self.resolution - 1)
            xs = coords[:, 0]
            ys = coords[:, 1]

            if DEBUG:

                np.add.at(temp_img_sum, (xs, ys), self.color_projector(X))
                np.add.at(temp_img_weights, (xs, ys), 1)

                assert np.all(temp_img_sum - temp_img_weights <= 0)

                self.img_sum += temp_img_sum
                self.img_weights += temp_img_weights

                temp_img_sum[:] = 0
                temp_img_weights[:] = 0
            else:
                np.add.at(self.img_sum, (xs, ys), self.color_projector(X))
                np.add.at(self.img_weights, (xs, ys), 1)

        tc = time.time() - tc
        print(f"{i=}, {tc=}")
        self.img = IMAGE_COMPILER(self.img_sum, self.img_weights)

        return self.img
