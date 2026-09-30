"""Small CPU checks for the custom gate, training state, and release catalog."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip("sae_lens")
import torch
from sae_lens import SAE
from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.training.sae_trainer import SAETrainer

from baselines.david.experiment import command_for, make_plan
from baselines.david.gates import MinFireGate
from baselines.david.recipes import BASE, recipe
from baselines.david.sae import build_sae
from baselines.david.train import (
    export_sae, gate_state, resample_dead, restore_checkpoint,
    save_checkpoint, set_gate_state,
)


def test_minfire_budget_rotation_and_gradient():
    x = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8).requires_grad_()
    gate = MinFireGate(k=2, every=2)
    for parity in [0, 1]:
        y = gate(x)
        assert (y > 0).sum() == 8
        assert (y[-1, parity::2] > 0).all()
        y.sum().backward()
        torch.testing.assert_close(x.grad, (y > 0).float())
        x.grad.zero_()
    gate.plain = True
    y = gate(x)
    assert torch.equal(y > 0, x > 24)


@pytest.mark.parametrize('name', [
    'btk_minfire_decay', 'mat_resample_win100_decay', 'jr_ste_c01_resample_decay',
])
def test_training_resume_resampling_and_export(name, tmp_path):
    torch.set_num_threads(1)
    r = recipe(name)
    r.update(autocast=False, mat_widths=(4, 16), dead_window=2)
    cfg = SAETrainerConfig(
        total_training_samples=64, train_batch_size_samples=8, device='cpu',
        autocast=False, lr=3e-4, lr_end=3e-4, lr_scheduler_name='constant',
        lr_decay_steps=2, dead_feature_window=2, n_checkpoints=0,
        save_final_checkpoint=False, logger=LoggingConfig(log_to_wandb=False),
    )

    def fresh():
        sae = build_sae(r, 4, 0, device='cpu', width=32, batch=8)
        trainer = SAETrainer(cfg, sae, data_provider=iter(()))
        trainer.activation_scaler.scaling_factor = 0.9 if r['arch'] == 'jr' else None
        return trainer

    def step(trainer):
        out = trainer.step(torch.randn(8, 768))
        trainer.n_training_steps += 1
        assert torch.isfinite(out.loss)

    trainer = fresh()
    step(trainer)
    metadata = {'recipe': r}
    extra = {'metadata': metadata}
    if r['gate'] != 'plain':
        extra['gate'] = gate_state(trainer.sae)
    save_checkpoint(tmp_path/'latest.pt', trainer, extra)
    step(trainer)
    expected = {k: v.clone() for k, v in trainer.sae.state_dict().items()}
    resumed = fresh()
    restored = restore_checkpoint(tmp_path/'latest.pt', resumed, metadata)
    if r['gate'] != 'plain':
        set_gate_state(resumed.sae, restored['gate'])
    step(resumed)
    for key, value in resumed.sae.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert resumed.n_training_samples == 16

    resumed.n_forward_passes_since_fired[:2] = 3
    resumed.n_forward_passes_since_fired[2:] = 0
    world = SimpleNamespace(sample=lambda n: torch.randn(n, 768))
    assert resample_dead(resumed.sae, resumed, world, r, 0, 2, 'cpu') == 2
    assert (resumed.n_forward_passes_since_fired[:2] == 0).all()
    for param, sl in [(resumed.sae.W_dec, (slice(0, 2),)),
                      (resumed.sae.W_enc, (slice(None), slice(0, 2)))]:
        assert torch.count_nonzero(resumed.optimizer.state[param]['exp_avg'][sl]) == 0
    export_sae(resumed.sae, resumed, tmp_path)
    inference = SAE.load_from_disk(tmp_path/'final', device='cpu').eval()
    resumed.sae.eval()
    x = torch.randn(8, 768)
    with torch.no_grad():
        # BatchTopK training encode retains its batch gate in eval mode; the
        # exported SAE instead uses the learned per-latent JumpReLU threshold.
        pre = resumed.sae.process_sae_in(resumed.activation_scaler(x)) @ resumed.sae.W_enc + resumed.sae.b_enc
        if getattr(resumed.sae.cfg, 'rescale_acts_by_decoder_norm', False):
            pre = pre * resumed.sae.W_dec.norm(dim=-1)
        threshold = resumed.sae.threshold if r['arch'] == 'jr' else resumed.sae.topk_threshold
        expected_codes = pre.relu() * (pre > threshold)
        torch.testing.assert_close(inference.encode(x), expected_codes, atol=2e-6, rtol=2e-6)
        assert torch.isfinite(inference.decode(inference.encode(x))).all()
    assert not any(k.startswith('autotuner.') for k in inference.state_dict())


def test_catalog_and_exact_experiment_plans():
    all_runs = make_plan('all')['runs']
    assert len(all_runs) == 209
    assert len(make_plan('screen')['runs']) == 162
    assert len(make_plan('headline')['runs']) == 42
    assert len({r['run_id'] for r in all_runs}) == 209
    for row in all_runs:
        assert row['stage'] in {'r1', 'r2', 'r3', 'grid', 'gridp'}
        assert row['train_samples'] == 200_000_000
        assert row['eval_samples'] == 1_000_000
        command = command_for(row, out_root=Path('results'), device='cpu')
        override = json.loads(command[command.index('--override')+1])
        effective = dict(recipe(row['recipe']), **override)
        for key, value in row['config']['recipe'].items():
            assert effective[key] == value
        # Early rounds predate these knobs; their original implicit defaults
        # are the same as BASE (l0_coef=1, revive=False).
        for key in set(effective) - set(row['config']['recipe']):
            assert effective[key] == BASE[key]
    low_l0 = [r for r in make_plan('headline')['runs'] if r['recipe']=='btk_minfire_decay' and r['target_l0']<30]
    assert {r['target_l0']:r['config']['recipe']['minfire_every'] for r in low_l0} == {15:5,20:3,25:2}
