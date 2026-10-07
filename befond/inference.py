"""Bernoulli posterior moments and natural-gradient inference.

The latent logits u parameterize q(z) = product_k Bernoulli(sigmoid(u_k)).
For a Gaussian likelihood, the natural gradient cancels the Bernoulli Fisher
factor r(1-r), leaving the simple recurrent update implemented below.
"""

import math

import torch
import torch.nn.functional as F


def bernoulli_moments(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E[z] and Var[z], retaining small variances at saturated logits."""
    mean = torch.exp(-F.softplus(-logits))
    complement = torch.exp(-F.softplus(logits))
    return mean, mean * complement


def bernoulli_kl(logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    """Elementwise KL(q || reference), evaluated in log space."""
    log_q1, log_q0 = -F.softplus(-logits), -F.softplus(logits)
    log_p1 = -F.softplus(-reference_logits)
    log_p0 = -F.softplus(reference_logits)
    return log_q1.exp() * (log_q1 - log_p1) + log_q0.exp() * (log_q0 - log_p0)


def predictive_logits(logits, prior_logits, retention=1.0, mix="logit"):
    """Mix the incoming belief with the learned baseline prior.

    Full retention gives the rolling posterior-as-prior rule. Zero retention
    anchors every outer step at the baseline prior. Intermediate retention
    interpolates either logits or probabilities.
    """
    if retention == 1.0:
        return logits
    if retention == 0.0:
        return prior_logits.expand_as(logits)
    if mix == "logit":
        return retention * logits + (1.0 - retention) * prior_logits
    if mix != "prob":
        raise ValueError("Prior mixing must be 'logit' or 'prob'.")
    log_keep, log_refresh = math.log(retention), math.log1p(-retention)
    log_p = torch.logaddexp(
        F.logsigmoid(logits) + log_keep,
        F.logsigmoid(prior_logits) + log_refresh,
    )
    log_complement = torch.logaddexp(
        F.logsigmoid(-logits) + log_keep,
        F.logsigmoid(-prior_logits) + log_refresh,
    )
    return log_p - log_complement


def inference_gain(integrator: str, h: float, beta: float) -> float:
    """Return the proposed gain; guarded ETD1 may reduce it by backtracking."""
    if h <= 0 or math.isnan(h) or beta < 0 or not math.isfinite(beta):
        raise ValueError("Inference requires a positive step and finite nonnegative beta.")
    if integrator == "euler":
        if not math.isfinite(h):
            raise ValueError("Euler requires a finite inference step.")
        return h
    if integrator not in ("etd1", "etd1_guarded"):
        raise ValueError("Inference integrator must be 'euler', 'etd1', or 'etd1_guarded'.")
    if math.isinf(h):
        if beta == 0:
            raise ValueError("Infinite ETD1 step requires beta > 0.")
        return 1.0 / beta
    return h if beta == 0 else -math.expm1(-beta * h) / beta


@torch.no_grad()
def guarded_inference_gain(model, x, logits, reference_logits, beta, direction, gain,
                           *, gram_diagonal=None, accept_step=None):
    """Backtrack ETD1 against every example's fixed-reference free energy.

    One scalar gain is shared by the batch. Try the proposal and at most twenty
    halvings, using the research implementation's Armijo and roundoff constants.
    Gaussian normalization terms are constant during inference and are omitted.
    """
    precision = model.variance.reciprocal()
    if gram_diagonal is None:
        gram_diagonal = (model.decoder.square() * precision.reshape(-1, 1)).sum(0)

    def energy(state):
        mean = state.sigmoid()
        variance = mean * (-state).sigmoid()
        residual = x - model.reconstruct(mean)
        reconstruction = 0.5 * (residual.square() * precision).sum(-1)
        uncertainty = 0.5 * (variance * gram_diagonal).sum(-1)
        kl = bernoulli_kl(state, reference_logits).sum(-1)
        return reconstruction + uncertainty + beta * kl

    before = energy(logits)
    variance = logits.sigmoid() * (-logits).sigmoid()
    descent = (variance * direction.square()).sum(-1)
    tolerance = 1e-6 * (1.0 + before.abs())
    for _ in range(21):
        after = energy(logits + gain * direction)
        accepted = torch.isfinite(after) & (
            after <= before - 1e-4 * gain * descent + tolerance)
        accepted = accepted.all()
        if accept_step is not None:
            accepted = accept_step(accepted)
        if bool(accepted):
            return gain
        gain *= 0.5
    raise RuntimeError("Inference free-energy backtracking failed after 20 halvings.")


def inference_direction(model, x, logits, reference_logits, beta=1.0,
                        *, drive=None, gram_diagonal=None):
    """Natural-gradient direction for Bernoulli logits, with a fixed reference.

    Phi is [D,K], r is [B,K], and Sigma is diagonal. The direction is
    Phi^T Sigma^-1 (x-b-Phi r) - diag(Phi^T Sigma^-1 Phi)(1-2r)/2
    - beta (u-u_ref). The matrix products avoid a dense K-by-K Gram matrix.
    """
    precision = model.variance.reciprocal()
    if drive is None:
        centered = x if model.bias is None else x - model.bias
        drive = (centered * precision) @ model.decoder
    if gram_diagonal is None:
        gram_diagonal = (model.decoder.square() * precision.reshape(-1, 1)).sum(0)
    mean = torch.exp(-F.softplus(-logits))
    complement = torch.exp(-F.softplus(logits))
    lateral = ((mean @ model.decoder.T) * precision) @ model.decoder
    direction = drive - lateral - 0.5 * gram_diagonal * (complement - mean)
    return direction - beta * (logits - reference_logits)
