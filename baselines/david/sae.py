"""SAELens model construction and the autotuned JumpReLU training loss."""
import dataclasses
import torch

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig
from sae_lens.saes.matryoshka_batchtopk_sae import (
    MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKTrainingSAEConfig,
)
from .coefficient_autotuner import CoefficientAutotuner, CoefficientAutotunerConfig
from .gates import MinFireGate, SinkhornGate

from .recipes import D_IN, D_SAE


class AutotunedJumpReLU(JumpReLUTrainingSAE):
    """SAELens 6.51 JumpReLU with the upstream sweep's L0-coefficient autotuner.

    Mirrors saes/jumprelu_sae.py of synth-sae-bench-experiments: the multiplier from the previous
    step's update scales the L0 coefficient of the current step; the buffers are stripped from the
    saved inference weights."""

    def __init__(self, cfg, tuner_cfg):
        super().__init__(cfg)
        self.autotuner = CoefficientAutotuner(tuner_cfg, device=cfg.device)

    def training_forward_pass(self, step_input):
        out = super().training_forward_pass(step_input)
        with torch.no_grad():
            batch_l0 = (out.feature_acts != 0).float().sum(dim=-1).mean()
        self.autotuner.update(batch_l0, step_input.n_training_steps)
        out.metrics['autotune_multiplier'] = self.autotuner.multiplier
        out.metrics['autotune_smoothed_l0'] = self.autotuner.smoothed_l0
        return out

    def calculate_aux_loss(self, step_input, feature_acts, hidden_pre, sae_out):
        coefficients = dict(step_input.coefficients)
        coefficients['l0'] = coefficients['l0'] * self.autotuner.multiplier
        scaled = dataclasses.replace(step_input, coefficients=coefficients)
        return super().calculate_aux_loss(scaled, feature_acts, hidden_pre, sae_out)

    def process_state_dict_for_saving_inference(self, state_dict):
        super().process_state_dict_for_saving_inference(state_dict)
        for key in [k for k in state_dict if k.startswith('autotuner.')]:
            del state_dict[key]


def minfire_period(r, k, batch, width):
    if r['minfire_every'] != 'auto':
        return int(r['minfire_every'])
    import math
    return max(1, math.ceil(width / (batch * k * 0.7)))


def build_sae(r, k, seed, device='cuda', width=D_SAE, batch=1024):
    torch.manual_seed(seed)
    if r['arch'] == 'btk':
        cfg = BatchTopKTrainingSAEConfig(d_in=D_IN, d_sae=width, k=k, device=device,
                                         aux_loss_coefficient=r['aux_coef'])
        sae = BatchTopKTrainingSAE(cfg)
    elif r['arch'] == 'mat':
        widths = [w for w in r['mat_widths'] if w < width] + [width]
        cfg = MatryoshkaBatchTopKTrainingSAEConfig(
            d_in=D_IN, d_sae=width, k=k, device=device, matryoshka_widths=widths,
            use_matryoshka_aux_loss=r['mat_aux'], aux_loss_coefficient=r['aux_coef'])
        sae = MatryoshkaBatchTopKTrainingSAE(cfg)
    elif r['arch'] == 'jr':
        cfg = JumpReLUTrainingSAEConfig(
            d_in=D_IN, d_sae=width, device=device,
            normalize_activations='expected_average_only_in',
            jumprelu_sparsity_loss_mode=r['penalty'], jumprelu_bandwidth=r['bw'],
            jumprelu_init_threshold=r['init_threshold'], jumprelu_tanh_scale=4.0,
            pre_act_loss_coefficient=r['preact'], decoder_init_norm=0.5,
            l0_coefficient=r['l0_coef'], l0_warm_up_steps=0, jumprelu_ste_to_input=r['ste'])
        tuner = CoefficientAutotunerConfig(target_l0=k, start_step=0, gain_scale=10.0,
                                           integral_gain=r['tuner_gain'], convergence_gain=1e-2)
        sae = AutotunedJumpReLU(cfg, tuner)
    else:
        raise ValueError(r['arch'])
    if r['gate'] == 'sinkhorn':
        sae.activation_fn = SinkhornGate(k, floor_frac=r['floor_frac'])
    elif r['gate'] == 'minfire':
        sae.activation_fn = MinFireGate(k, min_fires=r['min_fires'], every=minfire_period(r, k, batch, width))
    elif r['gate'] != 'plain':
        raise ValueError(r['gate'])
    return sae
