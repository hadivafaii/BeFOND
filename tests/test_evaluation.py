"""Evaluation changes inference settings without mutating learned weights."""

from pathlib import Path
from types import SimpleNamespace

import torch

from befond.config import load_config
from befond.model import BeFOND, ModelConfig
from experiments.gemma.evaluate import configure_inference


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_overrides_change_inference_without_changing_weights():
    torch.manual_seed(5)
    model = BeFOND(ModelConfig(input_dim=3, num_latents=4, inference_h=1.5,
                              inference_prior_retention=0.999))
    x = torch.randn(7, 3)
    original_config = model.config
    before = {key: value.clone() for key, value in model.state_dict().items()}
    cfg = load_config(ROOT / 'configs/gemma-512k-seed0.json')
    configure_inference(model, cfg['evaluation'])
    assert model.config.inference_h == float('inf')
    assert model.config.inference_prior_retention == 1
    assert original_config.inference_prior_retention == 0.999
    reference = BeFOND(ModelConfig(input_dim=3, num_latents=4, inference_h=float('inf'),
                                  inference_prior_retention=1))
    reference.load_state_dict(before)
    torch.testing.assert_close(model.infer(x, steps=10).mean,
                               reference.infer(x, steps=10).mean)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    # Older saved configurations without overrides preserve their model settings.
    old = SimpleNamespace(config=original_config)
    configure_inference(old, {'steps': 10})
    assert old.config == original_config

