from colorProjector import ColorProjector
from renderer import Renderer
from spaceProjector import InterpolatedSpaceProjector, ManifoldSpaceProjector
import spaceSampler


def main():
    n_dim = 5
    resolution = 6
    batch_size = 5000
    image_resolution = 2000

    X = spaceSampler.unit_grid(resolution, n_dim, hollow=True)
    space_projector = ManifoldSpaceProjector("isomap", n_dim, X)
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
    while True:
        renderer.render_step(100)
        renderer.update_plt()


if __name__ == "__main__":
    main()
