import time
import numpy as np
from renderOptions import RenderConfig
from colorProjector import ColorProjector
from renderer import Renderer
from spaceProjector import InterpolatedSpaceProjector, ManifoldSpaceProjector
import spaceSampler
from PIL import Image


class RenderPipeline(object):
    def __init__(self, render_config: RenderConfig):
        self.render_options = render_config

    def __call__(self):
        tc = time.time()

        opts = self.render_options
        n_dim = opts.n_dim
        resolution = opts.resolution
        batch_size = opts.batch_size
        image_resolution = opts.image_resolution

        X = spaceSampler.unit_grid(
            resolution, n_dim, hollow=opts.hollow_fit
        )
        space_projector = ManifoldSpaceProjector(opts.method, n_dim, X)
        interp_space_projector = InterpolatedSpaceProjector(
            resolution, n_dim, space_projector
        )
        X_proj = space_projector(X)
        print(f"{X.shape=}")
        print(f"{X_proj.shape=}")

        color_projector = ColorProjector(n_dim)
        color_arr = color_projector(X)

        print(f"{color_arr.shape=}")

        space_sampler1 = spaceSampler.unit_grid_iterator(
            resolution, n_dim, batch_size, sample_dim=1, edge_bias=20
        )
        space_sampler2 = spaceSampler.unit_grid_iterator(
            resolution, n_dim, batch_size, sample_dim=2, edge_bias=None
        )
        combo_sampler = spaceSampler.combo_iterator([space_sampler1, space_sampler2])

        renderer = Renderer(
            interp_space_projector, color_projector, combo_sampler, image_resolution
        )

        renderer.init_plt()
        while np.sum(renderer.img_weights) < opts.total_samples:
            renderer.render_step(100)
            renderer.update_plt()

        renderer.close_plt()

        img_uint8 = (renderer.img * 255).astype(np.uint8)
        img_path = opts.save_path + "_render.jpg"

        Image.fromarray(img_uint8).save(img_path)
        tc = time.time() - tc
        print(f"Rendered {opts.total_samples} samples in {tc=} seconds.")
