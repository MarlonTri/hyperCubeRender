from colorProjector import ColorProjector
from renderer import Renderer
from spaceProjector import InterpolatedSpaceProjector, ManifoldSpaceProjector
import spaceSampler


def main():
    n_dim = 4
    resolution = 12
    batch_size = 5000
    image_resolution = 2000

    X = spaceSampler.unit_grid(resolution, n_dim)
    space_projector = ManifoldSpaceProjector("isomap", n_dim, X)
    interp_space_projector = InterpolatedSpaceProjector(resolution, n_dim, space_projector)
    X_proj = space_projector(X)
    print(f"{X.shape=}")
    print(f"{X_proj.shape=}")

    color_projector = ColorProjector(n_dim)
    color_arr = color_projector(X)

    print(f"{color_arr.shape=}")

    space_sampler = spaceSampler.unit_grid_iterator(
        resolution, n_dim, batch_size 
    )

    renderer = Renderer(
        interp_space_projector, color_projector, space_sampler, image_resolution
    )

    renderer.init_plt()
    while True:
        renderer.render_step(100)
        renderer.update_plt()


if __name__ == "__main__":
    main()
