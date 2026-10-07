"""Expose recurrent BeFOND inference as an SAE with raw-coordinate I/O."""

import torch
from torch import nn

from befond.data.gemma_spec import MODEL_NAME, HOOK_NAME, HOOK_LAYER
from befond.data.normalization import ActivationNormalizer


class BeFONDAdapter(nn.Module):
    """Codes are posterior means rescaled by raw decoder norms.

    Decoder gradients remain available for RAVEL's intervention masks. The
    dictionary and iterative inference are frozen; there is no learned encoder.
    """

    def __init__(self, model, normalizer=None, *, steps=10, beta=1.0,
                 microbatch_size=256, readout="posterior_threshold", input_scale=1.0, cfg=None):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.normalizer = normalizer or ActivationNormalizer.none()
        self.steps, self.beta = int(steps), float(beta)
        self.input_scale = float(input_scale)
        if self.input_scale <= 0:
            raise ValueError("input_scale must be positive.")
        self.microbatch_size, self.readout = int(microbatch_size), readout
        if readout not in ("posterior_mean", "posterior_threshold"):
            raise ValueError("Readout must be posterior_mean or posterior_threshold.")
        decoder = self.normalizer.decoder_to_raw(model.decoder.detach() / self.input_scale).T
        norms = decoder.norm(dim=1)
        if (norms <= 0).any() or not torch.isfinite(norms).all():
            raise ValueError("SAEBench requires finite, nonzero decoder atoms.")
        bias = model.bias if model.bias is not None else decoder.new_zeros(decoder.shape[1])
        bias = self.normalizer.bias_to_raw(bias.expand(decoder.shape[1]) / self.input_scale)
        self.W_dec = nn.Parameter(decoder / norms[:, None], requires_grad=False)
        self.b_dec = nn.Parameter(bias.detach(), requires_grad=False)
        self.register_buffer("decoder_norms", norms)
        # Upstream compatibility fields; encoder diagnostics do not apply.
        self.W_enc = nn.Parameter(torch.zeros_like(decoder.T), requires_grad=False)
        self.b_enc = nn.Parameter(torch.zeros_like(norms), requires_grad=False)
        if cfg is None:
            from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig
            cfg = CustomSAEConfig(model_name=MODEL_NAME, hook_name=HOOK_NAME,
                                  hook_layer=HOOK_LAYER, d_in=decoder.shape[1], d_sae=len(decoder))
        self.cfg = cfg
        self.cfg.architecture = "befond_iterative"
        self.cfg.normalize_activations = "none"
        self._sync_cfg()

    @property
    def device(self):
        return self.W_dec.device

    @property
    def dtype(self):
        return self.W_dec.dtype

    def _sync_cfg(self):
        self.cfg.device = str(self.device)
        self.cfg.dtype = str(self.dtype).removeprefix("torch.")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._sync_cfg()
        return self

    @torch.inference_mode(False)
    @torch.no_grad()
    def encode(self, x):
        shape = (*x.shape[:-1], self.cfg.d_sae)
        flat = x.reshape(-1, self.cfg.d_in).to(device=self.device, dtype=self.dtype)
        result = self.W_dec.new_empty((len(flat), self.cfg.d_sae))
        for start in range(0, len(flat), self.microbatch_size):
            batch = self.normalizer.transform(flat[start:start + self.microbatch_size]) * self.input_scale
            means = self.model.infer(batch, steps=self.steps, beta=self.beta).mean
            if self.readout == "posterior_threshold":
                means = means * (means > 0.5)
            result[start:start + len(means)] = means * self.decoder_norms
        return result.reshape(shape)

    def decode(self, codes):
        return codes.to(device=self.device, dtype=self.dtype) @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))
