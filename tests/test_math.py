"""Small mathematical checks; no training data, service, or GPU is required."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from befond.inference import bernoulli_kl, bernoulli_moments, inference_direction
from befond.learning import project_prior_mean, update_parameters_, update_prior_
from befond.model import BeFOND, ModelConfig
from befond.solvers import decoder_update


def test_inference_matches_enumerated_bernoulli_free_energy():
    model = BeFOND(ModelConfig(input_dim=2, num_latents=3, fit_dec_bias=True)).double()
    x = torch.tensor([[0.7, -0.8], [0.2, 0.5]], dtype=torch.float64)
    logits = torch.tensor([[-1.2, 0.5, -0.3], [0.8, -0.4, 0.2]],
                          dtype=torch.float64, requires_grad=True)
    reference = model.initial_logits(2)
    states = torch.tensor(list(itertools.product((0.0, 1.0), repeat=3)), dtype=x.dtype)
    log_q = (states[None] * F.logsigmoid(logits[:, None])
             + (1 - states[None]) * F.logsigmoid(-logits[:, None])).sum(-1)
    residual = x[:, None] - model.reconstruct(states)[None]
    nll = 0.5 * (residual.square() / model.variance).sum(-1)
    beta = 0.7
    objective = (log_q.exp() * nll).sum() + beta * bernoulli_kl(logits, reference).sum()
    gradient, = torch.autograd.grad(objective, logits)
    _, variance = bernoulli_moments(logits)
    torch.testing.assert_close(inference_direction(model, x, logits, reference, beta),
                               -gradient / variance, rtol=1e-11, atol=1e-11)
    _, endpoint_variance = bernoulli_moments(torch.tensor([-60.0, 60.0]))
    assert (endpoint_variance > 0).all()


@pytest.mark.parametrize("metric", ["prior", "agg_predictive_prior", "factorized_predictive_prior",
                                    "agg_posterior", "factorized_posterior"])
@pytest.mark.parametrize("solver", ["dense", "matrix_free", "matrix_free_phi1",
                                    "matrix_free_phi1_chebyshev", "matrix_free_phi1_auto"])
def test_decoder_solvers_match_augmented_matrix_exponential(metric, solver):
    generator = torch.Generator().manual_seed(9)
    phi = torch.randn(3, 4, generator=generator, dtype=torch.float64) * 0.2
    x = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    mean = torch.rand(5, 4, generator=generator, dtype=torch.float64) * 0.5 + 0.1
    prior = torch.rand(1 if metric == "prior" else 5, 4,
                       generator=generator, dtype=torch.float64) * 0.3 + 0.1
    variance, prior_variance = mean * (1 - mean), prior * (1 - prior)
    metric_mean, metric_variance = prior, prior_variance
    if metric.endswith("posterior"):
        metric_mean, metric_variance = mean, variance
    if metric.startswith("factorized_"):
        metric_variance = metric_variance.mean(0, keepdim=True) + metric_mean.var(
            0, correction=0, keepdim=True)
        metric_mean = metric_mean.mean(0, keepdim=True)
    posterior = mean.T @ mean / len(mean) + torch.diag(variance.mean(0))
    metric_moment = (metric_mean.T @ metric_mean / len(metric_mean)
                     + torch.diag(metric_variance.mean(0)))
    operator = torch.linalg.solve(metric_moment, posterior)
    drive = torch.linalg.solve(metric_moment, mean.T @ x / len(x))
    augmented = torch.zeros(7, 7, dtype=phi.dtype)
    augmented[:4, :4], augmented[:4, 4:] = -operator, drive
    initial = torch.cat((phi.T, torch.eye(3, dtype=phi.dtype)))
    expected = (torch.linalg.matrix_exp(0.17 * augmented) @ initial)[:4].T
    actual, _ = decoder_update(phi=phi, x=x, mean=mean, variance=variance,
        prior_mean=prior, prior_variance=prior_variance, learning_rate=0.17,
        decoder_metric=metric, decoder_solver=solver,
        phi1_auto_chebyshev_tolerance=1e-10)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)


def test_prior_projection_and_parameter_updates():
    proposal = torch.tensor([0.9, 0.2, 0.01, 0.4], dtype=torch.float64)
    projected = project_prior_mean(proposal, budget=0.15, floor=0.03)
    assert projected.min() >= 0.03
    torch.testing.assert_close(projected.mean(), torch.tensor(0.15, dtype=proposal.dtype))
    free = projected > 0.03000001
    multiplier = ((proposal - projected) / (projected * (1 - projected)))[free]
    torch.testing.assert_close(multiplier, multiplier[0].expand_as(multiplier))
    model = BeFOND(ModelConfig(input_dim=2, num_latents=3, fit_dec_bias=True,
                               inference_h=0.5, t_outer=3)).double()
    x = torch.tensor([[0.3, -0.5], [0.2, 0.4]], dtype=torch.float64)
    result = model.infer(x)
    before = model.decoder.clone()
    diagnostics = update_parameters_(model, x, result, 0.03)
    assert not torch.equal(model.decoder, before)
    assert diagnostics["phi_delta_norm"] > 0
    assert model.variance.min() > 0
    update_prior_(model, result.mean.mean(0), 0.1, budget=0.02, floor=0.001)
    assert model.prior.mean() <= 0.0200000001
    assert all(not parameter.requires_grad for parameter in model.parameters())
