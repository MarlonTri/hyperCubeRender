from colorProjector import ColorProjector
from renderOptions import RenderConfig
from renderPipeline import RenderPipeline
from renderer import Renderer
from spaceProjector import InterpolatedSpaceProjector, ManifoldSpaceProjector
import spaceSampler

CONFIGS = [
    RenderConfig(
        n_dim=3,
        resolution=12,
        hollow_fit=False,
        method="isomap",
        total_samples=50_000_000,
        save_path="renders/test_3d",
    ),
    RenderConfig(
        n_dim=4,
        resolution=9,
        hollow_fit=True,
        method="isomap",
        total_samples=50_000_000,
        save_path="renders/test_4d_9rez",
    ),
    RenderConfig(
        n_dim=4,
        resolution=10,
        hollow_fit=True,
        method="isomap",
        total_samples=50_000_000,
        save_path="renders/test_4d_10rez",
    ),
    RenderConfig(
        n_dim=5,
        resolution=6,
        hollow_fit=True,
        method="isomap",
        total_samples=50_000_000,
        save_path="renders/test_5d",
    ),
]


def make_configs():
    configs = []

    for rez in range(3, 15):
        c = RenderConfig(
            n_dim=3,
            resolution=rez,
            hollow_fit=False,
            method="isomap",
            total_samples=50_000_000,
            save_path=f"renders/3D_{rez:02d}rez",
        )
        configs.append(c)

    for rez in range(3, 10):
        c = RenderConfig(
            n_dim=4,
            resolution=rez,
            hollow_fit=True,
            method="isomap",
            total_samples=50_000_000,
            save_path=f"renders/4D_{rez:02d}rez",
        )
        configs.append(c)

    for rez in range(3, 9):
        c = RenderConfig(
            n_dim=5,
            resolution=rez,
            hollow_fit=True,
            method="isomap",
            total_samples=50_000_000,
            save_path=f"renders/5D_{rez:02d}rez",
        )
        configs.append(c)

    return configs


def main():

    for config in make_configs():
        pipeline = RenderPipeline(config)
        pipeline(show_plt=False)


if __name__ == "__main__":
    main()
