class RenderOptions(object):
    def __init__(
        self,
        n_dim,
        resolution,
        hollow_fit,
        method,
        sampler_config,
        total_samples,
        save_path,
        batch_size=5000,
        image_resolution=2000,
    ):
        self.n_dim = n_dim
        self.resolution = resolution
        self.batch_size = batch_size
        self.image_resolution = image_resolution
        self.hollow_fit = hollow_fit
        self.method = method
        self.sampler_config = sampler_config
        self.total_samples = total_samples
        self.save_path = save_path
