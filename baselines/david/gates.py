"""Drop-in replacements for SAELens's `BatchTopK` activation that keep every latent firing.

Both are adapted from `synth_sae_bench/20_no_outside/sinkhorn_gate.py`. The mask is computed
in float32 (the encoder runs under bfloat16 autocast in the upstream recipe) and applied to
the ReLU codes. `plain` switches back to ordinary batch TopK for the closing stretch of
training so the trainer's running threshold recalibrates before the inference SAE is saved.
"""
import math

import torch
from torch import nn


def topk_count_mask(s, n):
    flat = s.flatten()
    n = int(max(min(n, flat.numel()), 0))
    m = torch.zeros_like(flat)
    if n > 0:
        m.scatter_(-1, torch.topk(flat, n, dim=-1).indices, 1.0)
    return m.reshape(s.shape)


def batch_topk_mask(s, k):
    return topk_count_mask(s, int(k * s.shape[0]))


@torch.no_grad()
def sinkhorn_mask(s, k, *, floor_frac=1.0, temperature=0.1, n_iters=5):
    """Batch TopK on codes nudged so every latent is owed `floor_frac` times its fair share
    `batch * k / n_latents` of the batch's firings; unioned with the plain mask."""
    B, G = s.shape
    share = B * k / G
    logK = s.clamp_min(1e-12).log() / temperature
    logK = logK - logK.mean(1, keepdim=True)
    log_w = (s / s.mean().clamp_min(1e-12)).clamp_min(1e-12).log()
    row, col = math.log(k), math.log(floor_frac * share)
    log_v = torch.zeros(1, G, device=s.device, dtype=s.dtype)
    for _ in range(n_iters):
        log_u = row - torch.logsumexp(logK + log_v, 1, keepdim=True)
        d = col - torch.logsumexp(logK + log_u + log_w, 0, keepdim=True)
        log_v = d.clamp_min(0.0)
    return torch.maximum(batch_topk_mask(logK + log_v, k), batch_topk_mask(s, k))


class SinkhornGate(nn.Module):
    def __init__(self, k, floor_frac=1.0, temperature=0.1, n_iters=5):
        super().__init__()
        self.k = k
        self.kw = dict(floor_frac=floor_frac, temperature=temperature, n_iters=n_iters)
        self.plain = False
        self.extra = 0.0      # EMA of extra gates per sample beyond k, for the log

    def forward(self, x):
        acts = x.relu()
        s = acts.reshape(-1, acts.shape[-1])
        if self.plain or not self.training:
            m = batch_topk_mask(s.float(), self.k)
        else:
            m = sinkhorn_mask(s.float(), self.k, **self.kw)
            self.extra = 0.99 * self.extra + 0.01 * float(m.sum() / s.shape[0] - self.k)
        return (s * m.to(s.dtype)).reshape(acts.shape)


class MinFireGate(nn.Module):
    """Batch TopK with a hard minimum: on every `every`-th batch a latent is forced to fire on its
    `min_fires` strongest samples (latent blocks take turns, so per batch only `n_latents / every`
    latents are forced). The remaining `k * batch - forced` budget is ordinary batch TopK, so the
    mean training L0 stays exactly k. A forced entry whose ReLU code is zero carries no gradient."""

    def __init__(self, k, min_fires=1, every=1):
        super().__init__()
        self.k = k
        self.min_fires = int(min_fires)
        self.every = int(every)
        self.plain = False
        self.t = 0
        self.extra = 0.0      # EMA of forced entries outside the plain top-k, per sample

    def forward(self, x):
        acts = x.relu()
        s = acts.reshape(-1, acts.shape[-1])
        B, G = s.shape
        budget = int(self.k * B)
        if self.plain or not self.training:
            m = topk_count_mask(s.float(), budget)
        else:
            with torch.no_grad():
                sf = s.float()
                sel = torch.arange(G, device=s.device) % self.every == self.t % self.every
                sub = sf[:, sel]
                idx = sub.topk(min(self.min_fires, B), dim=0).indices
                forced_sub = torch.zeros_like(sub).scatter_(0, idx, 1.0) * (sub > 0)
                forced = torch.zeros_like(sf)
                forced[:, sel] = forced_sub
                rest = sf.masked_fill(forced.bool(), float('-inf'))
                m = forced + topk_count_mask(rest, budget - int(forced.sum()))
                plain = topk_count_mask(sf, budget)
                self.extra = 0.99 * self.extra + 0.01 * float((forced * (1 - plain)).sum() / B)
                self.t += 1
        return (s * m.to(s.dtype)).reshape(acts.shape)
