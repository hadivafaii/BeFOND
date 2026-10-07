"""Guarded inference against the exact objective enumerated over latent states."""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from befond.inference import inference_direction, inference_gain, predictive_logits
from befond.model import BeFOND, ModelConfig


def enumerated_energy(model, x, logits, reference, beta):
    """Compute E_q[Gaussian energy + beta log(q/p)] without moment shortcuts."""
    states = torch.tensor(list(itertools.product((0.0, 1.0),
                          repeat=model.config.num_latents)), dtype=logits.dtype)
    log_q = (states[None] * F.logsigmoid(logits[:, None])
             + (1 - states[None]) * F.logsigmoid(-logits[:, None])).sum(-1)
    log_p = (states[None] * F.logsigmoid(reference[:, None])
             + (1 - states[None]) * F.logsigmoid(-reference[:, None])).sum(-1)
    residual = x[:, None] - model.reconstruct(states)[None]
    energy = 0.5 * (residual.square() / model.variance).sum(-1)
    return (log_q.exp() * (energy + beta * (log_q - log_p))).sum(-1)


def strong_decoder(mix="logit", h=1.5):
    model = BeFOND(ModelConfig(input_dim=2, num_latents=3, fit_dec_bias=True,
        inference_integrator="etd1_guarded", inference_h=h,
        inference_prior_mix=mix, inference_prior_retention=0.65)).double()
    model.decoder.copy_(torch.tensor([[8., 7., -6.], [5., -7., 5.]]))
    model.bias.copy_(torch.tensor([0.4, -0.2]))
    model.log_variance.copy_(torch.tensor([0.6, 1.3]).log())
    model.prior_logits.copy_(torch.tensor([-1.8, -2.1, -0.7]))
    x = torch.tensor([[0.7, -0.8], [3.2, 1.5]], dtype=torch.float64)
    logits = torch.tensor([[-1.2, 0.5, -0.3], [0.8, -0.4, 0.2]], dtype=x.dtype)
    return model, x, logits


@pytest.mark.parametrize("mix,beta,h,halvings", [
    ("logit", 0.7, 1.5, 4),
    ("prob", 0.0, 1.5, 5),
    ("prob", 0.7, math.inf, 5),
])
def test_guard_backtracks_against_enumerated_per_example_energy(mix, beta, h, halvings):
    model, x, logits = strong_decoder(mix, h)
    reference = predictive_logits(logits, model.prior_logits, 0.65, mix)
    initial = logits.clone().requires_grad_()
    before = enumerated_energy(model, x, initial, reference, beta)
    gradient, = torch.autograd.grad(before.sum(), initial)
    variance = initial.sigmoid() * (-initial).sigmoid()
    direction = -gradient / variance
    descent = (gradient.square() / variance).sum(-1)
    proposal = inference_gain("etd1", h, beta)
    expected_gain = proposal / 2 ** halvings

    drive, diagonal = model.inference_terms(x)
    result = model.inference_step(x, logits, beta=beta,
                                  drive=drive, gram_diagonal=diagonal)
    assert result.inference_gain == expected_gain
    torch.testing.assert_close(result.logits, logits + expected_gain * direction)
    torch.testing.assert_close(result.reference_logits, reference)
    after = enumerated_energy(model, x, result.logits, reference, beta)
    tolerance = 1e-6 * (1 + before.abs())
    assert torch.all(after <= before - 1e-4 * expected_gain * descent + tolerance)
    assert torch.any(enumerated_energy(model, x, logits + 2 * expected_gain * direction,
                      reference, beta) > before - 2e-4 * expected_gain * descent + tolerance)


def test_guard_requires_acceptance_for_every_example():
    model, x, logits = strong_decoder()
    # Four easy examples outweigh the one failing example in a batch mean.
    indices = torch.tensor([0, 0, 0, 0, 1])
    x, logits = x[indices], logits[indices]
    reference = predictive_logits(logits, model.prior_logits, 0.65, "logit")
    direction = inference_direction(model, x, logits, reference, beta=0.7)
    proposal = inference_gain("etd1", 1.5, 0.7)
    before = enumerated_energy(model, x, logits, reference, 0.7)
    rejected = enumerated_energy(model, x, logits + proposal / 8 * direction, reference, 0.7)
    assert rejected.mean() < before.mean()
    assert rejected[-1] > before[-1]
    result = model.inference_step(x, logits, beta=0.7)
    assert result.inference_gain == proposal / 16


def test_guard_accepts_ordinary_etd_proposal_when_it_decreases_energy():
    model = BeFOND(ModelConfig(input_dim=2, num_latents=3, inference_h=0.5)).double()
    x = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    logits = model.initial_logits(len(x))
    direction = inference_direction(model, x, logits, logits, beta=0.8)
    result = model.inference_step(x, logits, beta=0.8)
    assert result.inference_gain == inference_gain("etd1", 0.5, 0.8)
    torch.testing.assert_close(result.logits, logits + result.inference_gain * direction)


def test_guard_allows_twentieth_halving_but_has_a_finite_budget():
    model = BeFOND(ModelConfig(input_dim=1, num_latents=1, inference_h=1.5)).double()
    x = logits = torch.zeros(1, 1, dtype=torch.float64)
    model.decoder.fill_(150000)
    result = model.inference_step(x, logits, beta=0.0)
    assert result.inference_gain == 1.5 / 2 ** 20
    model.decoder.fill_(1000000)
    with pytest.raises(RuntimeError):
        model.inference_step(x, logits, beta=0.0)


def test_guard_rejects_nonfinite_observations():
    model = BeFOND(ModelConfig(input_dim=1, num_latents=1))
    with pytest.raises(RuntimeError):
        model.inference_step(torch.tensor([[math.nan]]), model.initial_logits(1))


@pytest.mark.parametrize("integrator", ["euler", "etd1"])
def test_original_integrators_preserve_multistep_recurrence(integrator):
    model = BeFOND(ModelConfig(input_dim=2, num_latents=3,
        inference_integrator=integrator, inference_h=0.5, t_inner=3,
        inference_prior_retention=0.6, inference_prior_mix="prob")).double()
    x = torch.tensor([[0.7, -0.8]], dtype=torch.float64)
    logits = torch.tensor([[-1.2, 0.5, -0.3]], dtype=x.dtype)
    reference = predictive_logits(logits, model.prior_logits, 0.6, "prob")
    gain = 0.5 if integrator == "euler" else -math.expm1(-0.8 * 0.5) / 0.8
    expected = logits.clone()
    for _ in range(3):
        expected = expected + gain * inference_direction(model, x, expected, reference, 0.8)
    result = model.inference_step(x, logits, beta=0.8)
    assert result.inner_steps == 3
    assert result.inference_gain == gain
    torch.testing.assert_close(result.logits, expected, rtol=0, atol=0)
