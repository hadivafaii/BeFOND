"""A tiny binary sparse-coding example that needs no external dataset."""

import torch
from .normalization import ActivationNormalizer


class ToySource:
    def __init__(self, config, device="cpu"):
        self.config = config["data"]
        self.device = torch.device(device)
        rng = torch.Generator(device=self.device).manual_seed(self.config.get("seed", 0))
        d, k = config["model"]["input_dim"], config["model"]["num_latents"]
        self.dictionary = torch.randn(d, k, generator=rng, device=self.device)
        self.dictionary *= 3 / self.dictionary.norm(dim=0)
        self.normalizer = ActivationNormalizer.none()
        self.normalization_path = None

    def sample(self, batch_size, step, stream="train"):
        offsets = {"train": 0, "prior": 300007, "validation": 100003, "test": 200003}
        rng = torch.Generator(device=self.device).manual_seed(
            self.config.get("seed", 0) + 1000003 * step + offsets[stream])
        codes = (torch.rand(batch_size, self.dictionary.shape[1], generator=rng,
                            device=self.device) < self.config.get("activity", 0.1)).float()
        noise = torch.randn(batch_size, self.dictionary.shape[0], generator=rng, device=self.device)
        return codes @ self.dictionary.T + self.config.get("noise", 0.1) * noise
