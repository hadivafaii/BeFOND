"""Natural-gradient learning of the dictionary, Gaussian noise, and prior.

Posterior statistics are held fixed during each parameter update. The decoder
follows a linear matrix ODE (solvers.py); the noise variance and prior means
follow scalar exponential relaxations. No gradient is backpropagated through
the inference trajectory.
"""

import math

import torch

from .inference import bernoulli_moments, inference_direction
from .solvers import decoder_update


def exponential_relaxation(current, target, rate, delta_t=1.0):
    """Exact solution of dy/dt = rate * (target - y) for a fixed target."""
    scale = float(rate) * float(delta_t)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Flow time must be finite and nonnegative.")
    return target + (current - target) * math.exp(-scale)


@torch.no_grad()
def project_prior_mean(proposal, budget=None, floor=0.0):
    """Forward-KL projection onto a mean activation budget and probability floor.

    Minimize sum_k KL(Ber(proposal_k) || Ber(p_k)) subject to p_k >= floor
    and mean(p) <= budget. A one-dimensional multiplier enforces the budget.
    Stored logits must remain finite, including when the user supplies floor=0.
    """
    if not math.isfinite(floor) or not 0.0 <= floor < 1.0:
        raise ValueError("Prior floor must be finite and in [0, 1).")
    if budget is not None and (
            not math.isfinite(budget) or budget <= 0.0 or floor > budget):
        raise ValueError("Prior budget must be finite, positive, and >= floor.")
    if not torch.isfinite(proposal).all() or ((proposal < 0) | (proposal > 1)).any():
        raise ValueError("Prior proposal must contain probabilities in [0, 1].")
    limits = torch.finfo(proposal.dtype)
    floor = max(float(floor), limits.tiny)
    ceiling = 1.0 - limits.eps / 2
    if floor > ceiling or (budget is not None and budget < floor):
        raise ValueError("Prior bounds cannot be represented by finite logits in this dtype.")
    probability = proposal.double()
    projected = probability.clamp(min=floor, max=ceiling)
    if budget is None or projected.mean().item() <= budget:
        return projected.to(proposal.dtype)
    if budget == floor:
        return torch.full_like(proposal, floor)

    def shifted(multiplier):
        # Stable quadratic root of the KKT stationarity equation.
        scale = 1.0 / (1.0 + multiplier)
        discriminant = ((1.0 - 2.0 * scale).square()
                        + 4.0 * scale * (1.0 - scale) * (1.0 - probability))
        return (2.0 * probability * scale / (1.0 + discriminant.sqrt())).clamp(
            min=floor, max=ceiling)

    lo, hi = probability.new_zeros(()), probability.max() / budget
    for _ in range(64):
        mid = (lo + hi) * 0.5
        over_budget = shifted(mid).mean() > budget
        lo, hi = torch.where(over_budget, mid, lo), torch.where(over_budget, hi, mid)
    return shifted(hi).to(proposal.dtype)


@torch.no_grad()
def update_prior_(model, target_mean, learning_rate, beta=1.0, delta_t=1.0,
                  budget=None, floor=0.0):
    """Fit the baseline prior to final posteriors from separate prior rollouts."""
    target_mean = target_mean.reshape_as(model.prior_logits)
    previous = model.prior_logits.clone()
    if model.config.fit_prior and learning_rate > 0 and beta > 0:
        proposal = exponential_relaxation(model.prior, target_mean,
                                          learning_rate * beta, delta_t).clamp_min(1e-12)
        mean = project_prior_mean(proposal, budget, floor)
        model.prior_logits.copy_(torch.logit(mean))
    return {"prior_delta_norm": (model.prior_logits - previous).norm().item(),
            "prior_rate_mean": model.prior.mean().item(),
            "prior_target_mean": target_mean.mean().item()}


@torch.no_grad()
def update_parameters_(model, x, result, learning_rate, *, variance_rate=None,
                       bias_rate=None, beta=1.0, delta_t=1.0,
                       decoder_metric="prior", decoder_solver="dense",
                       fisher_damping=0.0, chebyshev_degree=128,
                       phi1_auto_taylor_max_applications=32,
                       phi1_auto_chebyshev_tolerance=1e-5,
                       phi1_auto_chebyshev_max_degree=512):
    """Update decoder, optional bias, and noise from one final posterior.

    All proposals use the old parameters. The prior has a separate update so
    training can fit it with longer rollouts than the decoder. The bias statistic
    includes the response through one inference step, as in the research code.
    """
    variance_rate = learning_rate if variance_rate is None else variance_rate
    bias_rate = learning_rate if bias_rate is None else bias_rate
    old_decoder = model.decoder.detach()
    old_bias = model.bias.detach() if model.bias is not None else None
    centered = x if old_bias is None else x - old_bias
    prediction = result.mean @ old_decoder.T
    variance_target = ((centered - prediction).square()
                       + result.variance @ old_decoder.square().T).mean(0)
    if decoder_metric == "prior":
        prior_mean, prior_variance = bernoulli_moments(model.prior_logits.unsqueeze(0))
    else:
        prior_mean, prior_variance = result.reference_mean, result.reference_variance
    decoder_new, diagnostics = decoder_update(
        phi=old_decoder, x=centered, mean=result.mean, variance=result.variance,
        prior_mean=prior_mean, prior_variance=prior_variance,
        learning_rate=learning_rate, delta_t=delta_t, decoder_metric=decoder_metric,
        decoder_solver=decoder_solver, fisher_damping=fisher_damping,
        chebyshev_degree=chebyshev_degree,
        phi1_auto_taylor_max_applications=phi1_auto_taylor_max_applications,
        phi1_auto_chebyshev_tolerance=phi1_auto_chebyshev_tolerance,
        phi1_auto_chebyshev_max_degree=phi1_auto_chebyshev_max_degree,
        observation_precision=model.variance.reciprocal())
    bias_new = None
    if old_bias is not None:
        if result.inner_steps != 1:
            raise ValueError("Corrected bias learning requires one inner inference step.")
        direction = inference_direction(model, x, result.logits, result.reference_logits, beta)
        # This response statistic need not be a probability; do not clip it.
        response_mean = result.mean + result.inference_gain * result.variance * direction
        bias_residual = x - response_mean @ old_decoder.T
        if old_bias.numel() == 1:
            precision = model.variance.reciprocal().expand_as(bias_residual)
            target = (precision * bias_residual).sum() / precision.sum()
        else:
            target = bias_residual.mean(0)
        bias_new = exponential_relaxation(old_bias, target, bias_rate, delta_t)
    new_variance = None
    if model.config.fit_dec_var and variance_rate != 0:
        target = variance_target if model.log_variance.numel() > 1 else variance_target.mean()
        new_variance = exponential_relaxation(model.variance, target, variance_rate, delta_t)
        if not torch.isfinite(new_variance).all() or (new_variance <= 0).any():
            raise ValueError("Observation variance must stay finite and positive.")
    diagnostics.update(phi_delta_norm=(decoder_new - old_decoder).norm().item(),
                       bias_delta_norm=(0.0 if bias_new is None else
                                        (bias_new - old_bias).norm().item()),
                       var_target_mean=variance_target.mean().item(),
                       var_target_max=variance_target.max().item())
    model.decoder.copy_(decoder_new)
    if bias_new is not None:
        model.bias.copy_(bias_new)
    if new_variance is not None:
        model.log_variance.copy_(new_variance.log())
    return diagnostics
