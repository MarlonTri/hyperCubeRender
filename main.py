from colorProjector import ColorProjector
from renderOptions import RenderOptions
from renderPipeline import RenderPipeline
from renderer import Renderer
from spaceProjector import InterpolatedSpaceProjector, ManifoldSpaceProjector
import spaceSampler


def main():
    opts = RenderOptions(
        n_dim=3,
        resolution=12,
        hollow_fit=False,
        method="isomap",
        sampler_config=None,
        total_samples=1e7,
        save_path="renders/test_3d",
    )

    pipeline = RenderPipeline(opts)
    pipeline()
 
    opts = RenderOptions(
        n_dim=5,
        resolution=6,
        hollow_fit=True,
        method="isomap",
        sampler_config=None,
        total_samples=1e7,
        save_path="renders/test_5d",
    )


if __name__ == "__main__":
    main()
