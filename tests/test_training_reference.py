"""Optional end-to-end migration check using the original research Trainer.

Set FONDV2_REFERENCE to its checkout. Both trainers receive the same controlled
per-step data, so the comparison isolates inference, update order, schedules,
prior convergence, constraints, and decoder norm regularization.
"""

import json
import os
import sys

import pytest
import torch

import befond.training as public_training
from befond.config import resolve_config

REFERENCE = os.environ.get("FONDV2_REFERENCE")
pytestmark = pytest.mark.skipif(not REFERENCE, reason="optional research-reference comparison")


class ControlledSource:
    def sample(self, batch_size, step, stream="train"):
        offset = {"train": 0, "prior": 0.4, "validation": 0.8, "test": 1.2}[stream]
        coordinate = torch.arange(batch_size * 5, dtype=torch.float32).reshape(batch_size, 5)
        return 0.3 * torch.sin(coordinate * 0.17 + step * 0.31 + offset)


# Dense updates have independent matrix-exponential checks; retain a research
# comparison for capped-quadratic updates alongside the two prior-stop outcomes.
@pytest.mark.parametrize("init_scale,reject_threshold,solver,metric", [
    (None, 10.0, "dense", "prior"),
    (1.2, 1e-16, "capped_quadratic", "factorized_posterior"),
])
def test_training_matches_original_trainer(tmp_path, monkeypatch, init_scale, reject_threshold,
                                          solver, metric):
    sys.path.insert(0, REFERENCE)
    from fondv2.common.datafeeder import StationaryFeeder
    from fondv2.main.config_model import ConfigPoisVAE
    from fondv2.main.model import PoissonVAE
    from fondv2.train.config_train import ConfigTrain
    from fondv2.train.train_fondv2 import Trainer

    source = ControlledSource()

    class ControlledFeeder(StationaryFeeder):
        def __init__(self, stream, batch_size, max_steps):
            self.stream = stream
            super().__init__(x=None, batch_size=batch_size, max_steps=max_steps,
                             steps_per_cycle=max_steps, shuffle=False, pre_send_device=False)

        def prepare_step(self, step):
            return step

        def set_batch_data(self, iteration):
            self.reset()
            self.current_x = source.sample(self.batch_size, iteration, self.stream)[:, :, None, None]
            return self.current_x, None

    config = resolve_config({
        "model": {"input_dim": 5, "num_latents": 7, "seed": 10,
                  "init_scale": init_scale, "fit_dec_bias": True,
                  "inference_prior_retention": 0.999, "inference_h": 1.5,
                  "t_outer": 5, "t_inner": 1},
        "train": {"train_steps": 8, "batch_size": 5, "lr": 0.01, "lr_min": 0.002,
                  "horizon_dist": "exp-2", "warmup_portion": 0.125,
                  "ngd_lr_bias_mult": 0.7, "ngd_lr_sigma_mult": 0.5,
                  "ngd_lr_prior_mult": 0.8, "ngd_freeze_variance_after": 4,
                  "ngd_prior_update_every": 2, "ngd_prior_batch_size": 3,
                  "ngd_prior_n_batches": 2, "ngd_prior_t_outer": "auto",
                  "ngd_prior_t_outer_max": 12, "ngd_prior_kl_window": 2,
                  "ngd_prior_kl_threshold": [0.01, 0.005],
                  "ngd_prior_kl_reject_threshold": reject_threshold,
                  "ngd_decoder_solver": solver, "ngd_decoder_metric": metric,
                  "r0_budget": 0.03, "r0_floor": 0.015, "ngd_fisher_damping": 1e-6,
                  "decoder_norm_method": "soft", "decoder_norm_target": 0.4,
                  "decoder_norm_lamb_start": 0.5, "decoder_norm_lamb_end": 1.5,
                  "decoder_norm_lamb_anneal_portion": 0.625,
                  "eval_freq": 0, "chkpt_freq": 1, "log_freq": 1},
        "data": {"kind": "toy"}, "evaluation": {"steps": 4},
    })
    snapshots = {}

    def capture_checkpoint(model, path, *, step, **kwargs):
        snapshots[step] = {name: value.clone() for name, value in model.state_dict().items()}

    monkeypatch.setattr(public_training, "save_checkpoint", capture_checkpoint)
    public_training.train(config, tmp_path / "public", source=source)
    records = [json.loads(line) for line in (tmp_path / "public/metrics.jsonl").read_text().splitlines()]
    model_values = {key: value for key, value in config["model"].items()
                    if key not in ("input_dim", "num_latents")}
    original_model = PoissonVAE(ConfigPoisVAE(
        dataset="ActivationVectors5", latent_channels=7, learning_mode="ngd",
        inference_mode="exact", truncation_ceiling=1, **model_values))
    train_values = {key: value for key, value in config["train"].items()
                    if key not in ("stop_after_updates", "initial_decoder_norm")}
    train_values["eval_freq"] = 1000
    original_config = ConfigTrain(**train_values)
    trainer = Trainer(original_model, original_config, device="cpu", verbose=False,
                      feeders={"trn": ControlledFeeder("train", 5, 8),
                               "prior": ControlledFeeder("prior", 3, 8)})
    for step in range(8):
        trainer.train_step(step)
        expected = {"decoder": original_model.dec.weight.flatten(1).T,
                    "bias": original_model.dec.bias,
                    "log_variance": 2 * original_model.dec_log_sigma.flatten(),
                    "prior_logits": original_model.u_init[0].flatten()}
        for name, value in expected.items():
            torch.testing.assert_close(snapshots[step + 1][name], value,
                                       rtol=2e-5, atol=2e-6, msg=f"step {step}: {name}")
        assert records[step]["inference_steps"] == trainer.stats["ngd/horizon"][step]
        assert records[step]["lr"] == trainer.stats["lr"][step]
        if (step + 1) % 2 == 0:
            assert records[step]["prior_rejected"] == trainer.stats["ngd/prior_update_rejected"][step]
            assert records[step]["prior_inference_steps"] == trainer.stats["prior_stats/prior_t_outer"][step]
    assert any(record.get("prior_rejected") == int(reject_threshold < 1) for record in records)
