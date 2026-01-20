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
        total_samples=10_000_000,
        save_path="renders/test_3d",
    ),
    RenderConfig(
        n_dim=4,
        resolution=9,
        hollow_fit=True,
        method="isomap",
        total_samples=10_000_000,
        save_path="renders/test_4d",
    ),
    RenderConfig(
        n_dim=5,
        resolution=6,
        hollow_fit=True,
        method="isomap",
        total_samples=10_000_000,
        save_path="renders/test_5d",
    ),
]


def main():

    for config in CONFIGS:
        pipeline = RenderPipeline(config)
        pipeline()


if __name__ == "__main__":
    main()
