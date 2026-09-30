"""Training continuation and public artifact contracts, without external data."""

import json

import torch

from befond.checkpoints import convert_legacy, from_pretrained, save_pretrained
from befond.config import resolve_config
from befond.data import make_source
from befond.data.normalization import ActivationNormalizer
from befond.training import train


def configuration():
    return resolve_config({
        "model": {"input_dim": 5, "num_latents": 7, "seed": 2,
                  "fit_dec_var": True, "fit_dec_bias": True,
                  "t_outer": 3, "inference_h": 0.5},
        "train": {"train_steps": 6, "batch_size": 9, "lr": 0.01,
                  "lr_min": 0.001, "horizon_dist": "uniform",
                  "ngd_prior_update_every": 2, "ngd_prior_t_outer": 4,
                  "ngd_freeze_variance_after": 3, "r0_budget": 0.1,
                  "r0_floor": 1e-5, "eval_freq": 2, "eval_batch_size": 7,
                  "chkpt_freq": 2, "log_freq": 2},
        "data": {"kind": "toy", "seed": 3, "activity": 0.1, "noise": 0.05},
        "evaluation": {"steps": 4},
    })


def test_training_resume_and_portable_bundle(tmp_path):
    config = configuration()
    uninterrupted = train(config, tmp_path / "continuous")
    config["train"]["stop_after_updates"] = 3
    train(config, tmp_path / "resumed")
    config["train"]["stop_after_updates"] = None
    resumed = train(config, tmp_path / "resumed", resume=tmp_path / "resumed/checkpoint.pt")
    for name, tensor in uninterrupted.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[name], atol=0, rtol=0)
    loaded = from_pretrained(tmp_path / "resumed/pretrained")
    x = make_source(config).sample(11, 0, "test")
    torch.testing.assert_close(loaded.infer(x).logits, resumed.infer(x).logits, atol=0, rtol=0)
    normalizer = ActivationNormalizer.from_state_dict(torch.load(
        loaded.normalization_path, weights_only=True))
    torch.testing.assert_close(normalizer.transform(x), x)
    assert loaded.pretrained_metadata["step"] == 6


def test_legacy_conversion_preserves_width_precision_and_inference(tmp_path):
    from befond import BeFOND, ModelConfig

    model = BeFOND(ModelConfig(input_dim=5, num_latents=9, fit_dec_bias=True,
                               inference_h=0.5)).double()
    original = {"truncation_ceiling": 1, "learning_mode": "ngd", "latent_pixels": 1,
                "fit_dec_bias": True, "inference_h": 0.5}
    config_path = tmp_path / "ConfigPoisVAE.json"
    config_path.write_text(json.dumps(original))
    torch.save({"model": {"dec.weight": model.decoder.T[:, :, None, None],
                          "dec.bias": model.bias,
                          "dec_log_sigma": model.log_variance.reshape(5, 1, 1) / 2,
                          "u_init.0": model.prior_logits.reshape(1, 9, 1, 1)},
                "metadata": {"global_step": 5}}, tmp_path / "research.pt")
    convert_legacy(tmp_path / "research.pt", config_path, tmp_path / "bundle")
    loaded = from_pretrained(tmp_path / "bundle")
    x = torch.randn(4, 5, dtype=torch.float64)
    torch.testing.assert_close(loaded.infer(x).logits, model.infer(x).logits, atol=0, rtol=0)
    assert loaded.config.num_latents == 9
    assert loaded.decoder.dtype == torch.float64
