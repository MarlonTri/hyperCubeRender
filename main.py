from colorProjector import ColorProjector
from renderOptions import RenderConfig
from renderPipeline import RenderPipeline
from renderer import Renderer
from spaceProjector import InterpolatedSpaceProjector, ManifoldSpaceProjector
import spaceSampler


def make_configs():
    configs = []

    for rez in range(3, 15):
        c = RenderConfig(
            n_dim=3,
            resolution=rez,
            hollow_fit=False,
            method="isomap",
            total_samples=400_000_000,
            save_path=f"renders/3D_{rez:02d}rez",
        )
        configs.append(c)

    for rez in range(3, 10):
        c = RenderConfig(
            n_dim=4,
            resolution=rez,
            hollow_fit=True,
            method="isomap",
            total_samples=400_000_000,
            save_path=f"renders/4D_{rez:02d}rez",
        )
        configs.append(c)

    for rez in range(3, 9):
        c = RenderConfig(
            n_dim=5,
            resolution=rez,
            hollow_fit=True,
            method="isomap",
            total_samples=400_000_000,
            save_path=f"renders/5D_{rez:02d}rez",
        )
        configs.append(c)

    for rez in range(3, 9):
        c = RenderConfig(
            n_dim=6,
            resolution=rez,
            hollow_fit=True,
            method="isomap",
            total_samples=400_000_000,
            save_path=f"renders/6D_{rez:02d}rez",
        )
        configs.append(c)
    return configs


def main():

    for config in make_configs():
        pipeline = RenderPipeline(config)
        pipeline(show_plt=False)


if __name__ == "__main__":
    main()
