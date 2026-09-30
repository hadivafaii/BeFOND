"""Optional migration check against a local copy of the research repository.

Run FONDV2_REFERENCE=/path/to/research pytest tests/test_reference_parity.py.
The public package and normal test suite do not need that repository.
"""

import os
import sys

import pytest
import torch

from befond.learning import update_parameters_, update_prior_
from befond.model import BeFOND, ModelConfig

REFERENCE = os.environ.get("FONDV2_REFERENCE")
pytestmark = pytest.mark.skipif(not REFERENCE, reason="optional research-reference comparison")


@pytest.mark.parametrize("mix,retention,solver,metric", [
    ("logit", 1.0, "dense", "prior"),
    ("prob", 0.85, "matrix_free_phi1_auto", "agg_predictive_prior"),
    ("logit", 0.999, "capped_quadratic", "factorized_posterior"),
])
def test_bernoulli_trajectory_and_learning_match_reference(mix, retention, solver, metric):
    sys.path.insert(0, REFERENCE)
    from fondv2.main.config_model import ConfigPoisVAE
    from fondv2.main.model import PoissonVAE
    from fondv2.train.ngd import apply_vanilla_ngd_update_, update_initial_prior_

    config = ModelConfig(input_dim=5, num_latents=7, seed=8, init_scale=0.13,
                         inference_h=1.5, inference_prior_mix=mix,
                         inference_prior_retention=retention, fit_dec_bias=True)
    model = BeFOND(config).double()
    reference = PoissonVAE(ConfigPoisVAE(
        dataset="ActivationVectors5", latent_channels=7, seed=8, init_scale=0.13,
        truncation_ceiling=1, learning_mode="ngd", inference_mode="exact",
        inference_integrator="etd1", inference_h=1.5, inference_prior_mix=mix,
        inference_prior_retention=retention, fit_dec_bias=True, t_outer=4,
        t_inner=1, fit_prior=True, fit_dec_var=True, dec_var_pixelwise=True)).double()
    torch.testing.assert_close(model.decoder, reference.dec.weight.reshape(7, 5).T)
    torch.testing.assert_close(model.prior_logits, reference.u_init[0].flatten())
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    old_x = x[:, :, None, None]
    beta = 0.8
    logits = model.initial_logits(len(x))
    reference.reset_state(len(x))
    with torch.no_grad():
        for _ in range(4):
            result = model.inference_step(x, logits, beta=beta)
            old = reference.advance_dynamics(old_x, kl_beta=beta,
                                             capture_reference_moments=True)
            torch.testing.assert_close(result.logits, old["state"][0].flatten(1),
                                       rtol=1e-11, atol=1e-12)
            torch.testing.assert_close(result.mean, old["posterior"].mean.flatten(1),
                                       rtol=1e-11, atol=1e-12)
            logits = result.logits
        update_parameters_(model, x, result, 0.03, variance_rate=0.02,
                           bias_rate=0.01, beta=beta, decoder_solver=solver,
                           decoder_metric=metric, fisher_damping=1e-6)
        apply_vanilla_ngd_update_(reference, old_x, old["u_ref"], eta_phi=0.03,
                                 eta_sigma=0.02, eta_bias=0.01, kl_beta=beta,
                                 decoder_solver=solver, decoder_metric=metric,
                                 fisher_damping=1e-6, reference_moments=old["reference_moments"])
        torch.testing.assert_close(model.decoder, reference.dec.weight.reshape(7, 5).T,
                                   rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(model.bias, reference.dec.bias, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(model.variance, (2 * reference.dec_log_sigma).exp().flatten(),
                                   rtol=1e-10, atol=1e-12)
        target = result.mean.mean(0)
        update_prior_(model, target, 0.08, beta=beta, budget=0.02, floor=0.001)
        update_initial_prior_(reference, target, eta_prior=0.08, kl_beta=beta,
                              delta_t=1.0, r0_budget=0.02, r0_floor=0.001)
        torch.testing.assert_close(model.prior_logits, reference.u_init[0].flatten(),
                                   rtol=1e-10, atol=1e-12)
