import time
import numpy as np
from renderOptions import RenderConfig
from colorProjector import ColorProjector
from renderer import Renderer
from spaceProjector import InterpolatedSpaceProjector, ManifoldSpaceProjector
import spaceSampler
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


def add_overlay_top_left(
    img_np,
    text,
    pie_slices,
    pie_colors,
    margin=40,
    pie_size=300,
    text_color=(255, 255, 255, 255),
    font=None,
):
    """
    img_np: HxWx3 or HxWx4 numpy array
    """
    img = Image.fromarray(img_np.astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    if font is None:
        mfont = font_manager.FontProperties(family="Courier New", weight="bold")
        file = font_manager.findfont(mfont)
        font = ImageFont.truetype(
            file,
            size=48,
        )

    # ---- text ----
    text_pos = (margin, margin)
    draw.text(text_pos, text, fill=text_color, font=font)

    # estimate text height to place pie below it
    bbox = draw.textbbox(text_pos, text, font=font)
    text_h = bbox[3] - bbox[1]

    # ---- pie chart ----
    pie_top = margin + text_h + 20
    pie_bbox = (
        margin,
        pie_top,
        margin + pie_size,
        pie_top + pie_size,
    )
    pie_colors = (pie_colors * 255).astype(int).tolist()
    pie_colors = [tuple(c) for c in pie_colors]
    for i, c in enumerate(pie_colors):
        pie_angle = 360 / pie_slices
        start, end = i * pie_angle, (i + 1) * pie_angle
        draw.pieslice(pie_bbox, start, end, fill=c)

    return np.array(img)


class RenderPipeline(object):
    def __init__(self, render_config: RenderConfig):
        self.render_options = render_config

    def __call__(self, show_plt):

        print("Starting pipeline:", self.render_options)
        tc = time.time()

        opts = self.render_options
        n_dim = opts.n_dim
        resolution = opts.resolution
        batch_size = opts.batch_size
        image_resolution = opts.image_resolution

        X = spaceSampler.unit_grid(resolution, n_dim, hollow=opts.hollow_fit)
        space_projector = ManifoldSpaceProjector(opts.method, n_dim, X)
        interp_space_projector = InterpolatedSpaceProjector(
            resolution, n_dim, space_projector
        )

        color_projector = ColorProjector(n_dim)

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

        if show_plt:
            renderer.init_plt()
        while np.sum(renderer.img_weights) < opts.total_samples:
            renderer.render_step(100)
            if show_plt:
                renderer.update_plt()

        if show_plt:
            renderer.close_plt()

        img_uint8 = (renderer.img * 255).astype(np.uint8)
        image_text = (
            f"DIM={n_dim}\nMESH-RES={resolution}\nSAMPLES={opts.total_samples:.1e}"
        )
        img_uint8 = add_overlay_top_left(
            img_uint8, image_text, n_dim, color_projector.colors[:-1]
        )
        img_path = opts.save_path + "_render.png"

        Image.fromarray(img_uint8).save(img_path)
        tc = time.time() - tc
        print(f"Rendered {opts.total_samples} samples in {tc=} seconds.")
