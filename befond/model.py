"""A Bernoulli latent model with a linear Gaussian decoder.

Read inference.py for the posterior dynamics and learning.py for parameter
updates. This class stores the generative model and provides a small tensor
interface; it has no learned encoder and does not build autograd trajectories.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .inference import bernoulli_kl, bernoulli_moments, inference_direction
from .inference import guarded_inference_gain, inference_gain, predictive_logits


@dataclass
class ModelConfig:
    input_dim: int
    num_latents: int
    seed: int = 0
    init_scale: float | None = None
    prior_init_lower: float = -4.0
    prior_init_upper: float = -3.0
    fit_prior: bool = True
    fit_dec_bias: bool = False
    dec_bias_pixelwise: bool = True
    fit_dec_var: bool = True
    dec_var_pixelwise: bool = True
    inference_integrator: str = "etd1_guarded"
    inference_h: float = math.inf
    inference_prior_mix: str = "logit"
    inference_prior_retention: float = 1.0
    t_outer: int = 100
    t_inner: int = 1

    def __post_init__(self):
        self.inference_h = float(self.inference_h)
        if self.input_dim < 1 or self.num_latents < 1:
            raise ValueError("Input dimension and number of latents must be positive.")
        if self.t_outer < 1 or self.t_inner < 1:
            raise ValueError("Inference step counts must be positive.")
        if self.inference_integrator == "etd1_guarded" and self.t_inner != 1:
            raise ValueError("etd1_guarded requires t_inner=1.")
        if not 0.0 <= self.inference_prior_retention <= 1.0:
            raise ValueError("Prior retention must be between zero and one.")
        if self.inference_prior_mix not in ("logit", "prob"):
            raise ValueError("Prior mixing must be 'logit' or 'prob'.")
        if self.prior_init_lower > self.prior_init_upper:
            raise ValueError("Prior initialization bounds are reversed.")
        if self.init_scale is not None and self.init_scale <= 0:
            raise ValueError("Decoder initialization scale must be positive.")
        inference_gain(self.inference_integrator, self.inference_h, 1.0)


@dataclass
class InferenceResult:
    """Final posterior, fixed reference, and accepted gain of its last step."""
    logits: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    reference_logits: torch.Tensor
    reference_mean: torch.Tensor
    reference_variance: torch.Tensor
    inference_gain: float
    inner_steps: int


class BeFOND(nn.Module):
    """Factorized Bernoulli prior and p(x|z) = N(b + Phi z, diagonal variance).

    Observations have shape [batch, input_dim]; posterior moments have shape
    [batch, num_latents]. The decoder columns are the dictionary features.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        scale = 0.1 if config.init_scale is None else config.init_scale
        weights = torch.randn(config.num_latents, config.input_dim, generator=generator)
        if config.init_scale is not None:
            # Match the explicit reinitialization used by the experiment recipes.
            weights = torch.randn(config.num_latents, config.input_dim, generator=generator)
        self.decoder = nn.Parameter((scale * weights).T.contiguous(), requires_grad=False)
        bias = (torch.zeros(config.input_dim if config.dec_bias_pixelwise else 1)
                if config.fit_dec_bias else None)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        size = config.input_dim if config.dec_var_pixelwise else 1
        self.log_variance = nn.Parameter(torch.zeros(size), requires_grad=False)
        prior = np.random.default_rng(config.seed).uniform(
            config.prior_init_lower, config.prior_init_upper, config.num_latents)
        self.prior_logits = nn.Parameter(torch.tensor(prior, dtype=torch.float32),
                                         requires_grad=False)

    @property
    def variance(self):
        """Observation variance, shared or one value per observed coordinate."""
        return self.log_variance.exp()

    @property
    def prior(self):
        """Baseline Bernoulli activation probabilities."""
        return bernoulli_moments(self.prior_logits)[0]

    def initial_logits(self, batch_size):
        return self.prior_logits.unsqueeze(0).expand(batch_size, -1)

    def reconstruct(self, mean):
        prediction = mean @ self.decoder.T
        return prediction if self.bias is None else prediction + self.bias

    def inference_terms(self, x):
        """Cache these two terms when x and model parameters stay fixed."""
        precision = self.variance.reciprocal()
        centered = x if self.bias is None else x - self.bias
        drive = (centered * precision) @ self.decoder
        diagonal = (self.decoder.square() * precision.reshape(-1, 1)).sum(0)
        return drive, diagonal

    @torch.no_grad()
    def inference_step(self, x, logits, *, beta=1.0, inner_steps=None,
                       drive=None, gram_diagonal=None, accept_step=None):
        """Advance one outer step with a fixed predictive reference.

        ``accept_step`` optionally reduces the guard's scalar acceptance flag
        across workers inferring shards of the same batch. Ordinary inference
        needs no callback, including evaluation on just one worker.
        """
        if x.ndim != 2 or x.shape[1] != self.config.input_dim:
            raise ValueError("Observations must have shape [batch, input_dim].")
        if logits.shape != (len(x), self.config.num_latents):
            raise ValueError("Logits must have shape [batch, num_latents].")
        inner_steps = self.config.t_inner if inner_steps is None else int(inner_steps)
        if inner_steps < 1:
            raise ValueError("inner_steps must be positive.")
        if self.config.inference_integrator == "etd1_guarded" and inner_steps != 1:
            raise ValueError("etd1_guarded requires inner_steps=1.")
        reference = predictive_logits(logits, self.prior_logits,
            self.config.inference_prior_retention, self.config.inference_prior_mix)
        gain = inference_gain(self.config.inference_integrator, self.config.inference_h, beta)
        reference_mean, reference_variance = bernoulli_moments(reference)
        for _ in range(inner_steps):
            direction = inference_direction(
                self, x, logits, reference, beta,
                drive=drive, gram_diagonal=gram_diagonal)
            if self.config.inference_integrator == "etd1_guarded":
                gain = guarded_inference_gain(
                    self, x, logits, reference, beta, direction, gain,
                    gram_diagonal=gram_diagonal, accept_step=accept_step)
            logits = logits + gain * direction
        mean, variance = bernoulli_moments(logits)
        return InferenceResult(logits, mean, variance, reference,
                               reference_mean, reference_variance, gain, inner_steps)

    @torch.no_grad()
    def infer(self, x, steps=None, beta=1.0, inner_steps=None, *, accept_step=None):
        """Reset each input to the baseline prior and run recurrent inference."""
        steps = self.config.t_outer if steps is None else int(steps)
        if steps < 1:
            raise ValueError("steps must be positive.")
        logits = self.initial_logits(len(x))
        drive, diagonal = self.inference_terms(x)
        for _ in range(steps):
            result = self.inference_step(x, logits, beta=beta, inner_steps=inner_steps,
                                         drive=drive, gram_diagonal=diagonal,
                                         accept_step=accept_step)
            logits = result.logits
        return result

    def forward(self, x, steps=None, beta=1.0, inner_steps=None):
        return self.infer(x, steps=steps, beta=beta, inner_steps=inner_steps)

    def loss(self, x, result: InferenceResult, beta=1.0):
        """Exact per-example expected Gaussian NLL, KL, free energy and MSE."""
        residual = x - self.reconstruct(result.mean)
        uncertainty = result.variance @ self.decoder.square().T
        nll = 0.5 * ((residual.square() + uncertainty) / self.variance
                     + self.log_variance + math.log(2.0 * math.pi)).sum(-1)
        kl = bernoulli_kl(result.logits, result.reference_logits).sum(-1)
        return {"reconstruction": nll, "kl": kl, "free_energy": nll + beta * kl,
                "mse": residual.square().mean(-1)}
