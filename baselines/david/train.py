"""Train and score one full-width SAE on the released SynthSAEBench 16k-v1 world.

Native SAELens optimisation steps; a local loop supplies the schedules (gate window, bandwidth
anneal, resampling), atomic resumable checkpoints and logs. Scoring uses the unmodified SAELens
synthetic scorer on the saved ordinary inference SAE (JumpReLU gate, native thresholds).
"""
import argparse
import contextlib
import copy
import json
import math
import pathlib
import signal
import time

from .recipes import D_SAE, recipe
from .sae import build_sae

import torch
from huggingface_hub import snapshot_download
from sae_lens import SAE, __version__ as saelens_version
from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.synthetic import SyntheticModel, eval_sae_on_synthetic_data
from sae_lens.synthetic.training import SyntheticActivationIterator
from sae_lens.training.sae_trainer import SAETrainer

MODEL_REPO = 'decoderesearch/synth-sae-bench-16k-v1'
MODEL_REVISION = 'b2efd8b919ae46d6d487c73d46db5ee52813621d'
OUT_ROOT = pathlib.Path('outputs/david')
TRAINER_FIELDS = ['n_training_samples', 'n_training_steps', 'act_freq_scores',
                  'n_forward_passes_since_fired', 'n_frac_active_samples', 'started_fine_tuning']


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=float) + '\n')
    tmp.replace(path)


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


@contextlib.contextmanager
def forked_rng(seed, device):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def bandwidth_at(frac, start, end, until):
    progress = min(max(frac / until, 0.0), 1.0)
    return start * (end / start) ** progress


@torch.no_grad()
def calibrate_threshold(sae, trainer, world, k, seed, device):
    """Set the shared initial threshold so that the initial L0 on a 512-observation draw is k."""
    with forked_rng(2000 + seed, device):
        x = trainer.activation_scaler(world.sample(512).to(device))
        pre = sae.process_sae_in(x) @ sae.W_enc + sae.b_enc
        threshold = pre.flatten().topk(int(k * len(x))).values[-1].clamp_min(1e-3)
        sae.threshold.fill_(float(threshold))
        return float(threshold)


@torch.no_grad()
def resample_dead(sae, trainer, world, r, seed, step, device):
    """Anthropic-style resampling: dead latents restart on residual directions of badly
    reconstructed inputs, with a small encoder, zero bias, reset Adam moments and a fresh
    inactivity counter."""
    dead = trainer.n_forward_passes_since_fired > r['dead_window']
    n = int(dead.sum())
    if n == 0:
        return 0
    live = ~dead
    was_training = sae.training
    sae.eval()
    with forked_rng(700_000 + seed * 1000 + step % 1000, device):
        x = trainer.activation_scaler(world.sample(8192).to(device)).float()
    resid = x - sae.decode(sae.encode(x))
    p = resid.square().sum(1)
    idx = torch.multinomial(p / p.sum(), n, replacement=True)
    dirs = torch.nn.functional.normalize(resid[idx], dim=1)
    dec_norm = sae.W_dec[live].norm(dim=1).mean() if live.any() else 1.0
    enc_norm = 0.2 * (sae.W_enc[:, live].norm(dim=0).mean() if live.any() else 1.0)
    sae.W_dec.data[dead] = (dirs * dec_norm).to(sae.W_dec.dtype)
    sae.W_enc.data[:, dead] = (dirs * enc_norm).T.to(sae.W_enc.dtype)
    sae.b_enc.data[dead] = 0.0
    slices = {sae.W_dec: (dead,), sae.W_enc: (slice(None), dead), sae.b_enc: (dead,)}
    if hasattr(sae, 'threshold'):
        own_pre = enc_norm * resid[idx].norm(dim=1)     # pre-activation on its own source sample
        sae.threshold.data[dead] = (0.5 * own_pre).clamp_min(1e-3).to(sae.threshold.dtype)
        slices[sae.threshold] = (dead,)
    for param, sl in slices.items():
        state = trainer.optimizer.state.get(param)
        if state:
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key][sl] = 0.0
    trainer.n_forward_passes_since_fired[dead] = 0
    if was_training:
        sae.train()
    return n


def gate_state(sae):
    g = sae.activation_fn
    return dict(plain=g.plain, extra=g.extra, t=getattr(g, 't', 0))


def set_gate_state(sae, state):
    g = sae.activation_fn
    g.plain, g.extra = state['plain'], state['extra']
    if hasattr(g, 't'):
        g.t = state['t']


def save_checkpoint(path, trainer, extra):
    state = dict(model=trainer.sae.state_dict(), optimizer=trainer.optimizer.state_dict(),
                 lr_scheduler=trainer.lr_scheduler.state_dict(),
                 coefficients={k: v.state_dict() for k, v in trainer.coefficient_schedulers.items()},
                 trainer={k: getattr(trainer, k) for k in TRAINER_FIELDS},
                 scaling_factor=trainer.activation_scaler.scaling_factor, extra=extra,
                 cpu_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(trainer.cfg.device) if torch.device(trainer.cfg.device).type == 'cuda' else None)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def restore_checkpoint(path, trainer, metadata):
    saved = torch.load(path, map_location=trainer.cfg.device, weights_only=False)
    if saved['extra']['metadata'] != metadata:
        raise ValueError('Checkpoint settings changed; use a new run directory.')
    trainer.sae.load_state_dict(saved['model'])
    trainer.optimizer.load_state_dict(saved['optimizer'])
    trainer.lr_scheduler.load_state_dict(saved['lr_scheduler'])
    for k, v in saved['coefficients'].items():
        trainer.coefficient_schedulers[k].load_state_dict(v)
    for k, v in saved['trainer'].items():
        setattr(trainer, k, v)
    trainer.activation_scaler.scaling_factor = saved['scaling_factor']
    torch.set_rng_state(saved['cpu_rng'].cpu())
    if saved['cuda_rng'] is not None and torch.device(trainer.cfg.device).type == 'cuda':
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), trainer.cfg.device)
    return saved['extra']


@torch.no_grad()
def export_sae(sae, trainer, folder):
    export = copy.deepcopy(sae)
    if trainer.activation_scaler.scaling_factor is not None:
        export.fold_activation_norm_scaling_factor(trainer.activation_scaler.scaling_factor)
    if hasattr(export, 'threshold'):
        export.threshold.data.clamp_(min=0.0)
    export.save_inference_model(str(folder / 'final'))
    del export


@torch.no_grad()
def score_saved(world, args, folder):
    inference = SAE.load_from_disk(str(folder / 'final'), device=args.device).eval()
    torch.set_float32_matmul_precision('highest')
    with forked_rng(49100 + args.seed, args.device):
        r = eval_sae_on_synthetic_data(inference, world.feature_dict, world.activation_generator,
                                       num_samples=args.eval_samples, batch_size=4096)
    torch.set_float32_matmul_precision('high')
    return dict(mcc=r.mcc, f1=r.classification.f1_score, precision=r.classification.precision,
                recall=r.classification.recall, r2=r.explained_variance, sae_l0=r.sae_l0,
                true_l0=r.true_l0, dead=r.dead_latents, uniqueness=r.uniqueness, shrinkage=r.shrinkage)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--k', type=float, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--samples', type=int, default=200_000_000)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-samples', type=int, default=1_000_000)
    ap.add_argument('--tag', default='dev')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--width', type=int, default=D_SAE)
    ap.add_argument('--norm-batches', type=int, default=1000)
    ap.add_argument('--stop-after-steps', type=int, default=0)
    ap.add_argument('--skip-eval', action='store_true')
    ap.add_argument('--checkpoint-minutes', type=float, default=10.0)
    ap.add_argument('--override', default='{}', help='JSON dict of recipe knobs to override (ad-hoc variants)')
    ap.add_argument('--suffix', default='', help='run-folder suffix for ad-hoc variants')
    ap.add_argument('--out-root', default=str(OUT_ROOT), help='results directory')
    ap.add_argument('--offline', action='store_true', help='use only the cached benchmark world')
    ap.add_argument('--world-path', type=pathlib.Path, help='local copy of the pinned benchmark world')
    args = ap.parse_args()
    device_type = torch.device(args.device).type
    if args.samples <= 0 or args.batch <= 0 or args.k <= 0 or args.width <= 0:
        ap.error('samples, batch, k, and width must be positive')
    if device_type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No CUDA device; use --device cpu or select an available CUDA device.')
    torch.set_num_threads(4 if device_type == 'cuda' else 1)
    torch.set_float32_matmul_precision('high')
    r = recipe(args.recipe)
    override = json.loads(args.override)
    unknown = set(override) - set(r)
    if unknown:
        raise KeyError(f'unknown recipe knobs {unknown}')
    r.update(override)
    k = int(args.k) if float(args.k).is_integer() else args.k
    folder = pathlib.Path(args.out_root) / args.tag / f'{args.recipe}{args.suffix}_k{k}_s{args.seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'score.json').exists():
        print('Already scored', folder, flush=True)
        return

    world_path = args.world_path or snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=args.offline,
                                   allow_patterns=['synthetic_model_config.json', 'synthetic_model.safetensors', 'hierarchy.json'])
    world_cfg = json.loads((pathlib.Path(world_path) / 'synthetic_model_config.json').read_text())
    world = SyntheticModel.load_from_disk(world_path, device=args.device)
    metadata = dict(recipe=r, k=k, seed=args.seed, samples=args.samples, batch=args.batch, lr=args.lr,
                    width=args.width, model_repo=MODEL_REPO, model_revision=MODEL_REVISION,
                    scale_children_by_parent=find_key(world_cfg, 'scale_children_by_parent'),
                    saelens=dict(version=saelens_version, revision='b5711e34d072846dc112881f4c8d209407a32b56'),
                    norm_batches=args.norm_batches)
    steps_total = math.ceil(args.samples / args.batch)

    sae = build_sae(r, k, args.seed, device=args.device, width=args.width, batch=args.batch)
    data = SyntheticActivationIterator(world.feature_dict, world.activation_generator, args.batch,
                                       autocast=r['autocast'])
    trainer_cfg = SAETrainerConfig(
        total_training_samples=args.samples, train_batch_size_samples=args.batch, device=args.device,
        autocast=r['autocast'], lr=args.lr, lr_end=args.lr, lr_scheduler_name='constant',
        lr_warm_up_steps=0, lr_decay_steps=int(r['lr_decay'] * steps_total), n_restart_cycles=1,
        adam_beta1=0.9, adam_beta2=0.999, dead_feature_window=r['dead_window'],
        feature_sampling_window=2000, n_checkpoints=0, save_final_checkpoint=False,
        n_batches_for_norm_estimate=args.norm_batches, logger=LoggingConfig(log_to_wandb=False))
    trainer = SAETrainer(cfg=trainer_cfg, sae=sae, data_provider=data)
    checkpoint = folder / 'latest.pt'
    if checkpoint.exists():
        extra = restore_checkpoint(checkpoint, trainer, metadata)
        if r['gate'] != 'plain':
            set_gate_state(sae, extra['gate'])
        print('Resumed at step', trainer.n_training_steps, flush=True)
    else:
        torch.manual_seed(100101 + args.seed)
        init = {}
        if sae.cfg.normalize_activations == 'expected_average_only_in':
            trainer.activation_scaler.estimate_scaling_factor(
                d_in=sae.cfg.d_in, data_provider=data, n_batches_for_norm_estimate=args.norm_batches)
            init['scaling_factor'] = trainer.activation_scaler.scaling_factor
        if r['arch'] == 'jr':
            with torch.no_grad(), forked_rng(2000 + args.seed, args.device):
                x = trainer.activation_scaler(world.sample(512).to(args.device))
                pre = sae.process_sae_in(x) @ sae.W_enc + sae.b_enc
                init['initial_pre_std'] = float(pre.std())
                init['initial_l0'] = float((pre > sae.threshold).sum(1).float().mean())
            if r['calibrate_threshold']:
                init['calibrated_threshold'] = calibrate_threshold(sae, trainer, world, k, args.seed, args.device)
        extra = dict(metadata=metadata, init=init, history=[], train_seconds=0.0, resampled=0)
    write_json(folder / 'run.json', dict(metadata, init=extra['init']))

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append('TERM'))
    signal.signal(signal.SIGUSR1, lambda *_: stop.append('USR1'))
    last_save = time.monotonic()
    interval = max(1, steps_total // 100)
    while trainer.n_training_samples < args.samples:
        start = time.monotonic()
        frac = trainer.n_training_samples / args.samples
        if r['arch'] == 'jr' and r['bw_end'] is not None:
            sae.bandwidth = bandwidth_at(frac, r['bw'], r['bw_end'], r['bw_until'])
        if r['gate'] != 'plain':
            sae.activation_fn.plain = frac >= r['gate_until']
        trainer.maybe_reset_sparsity()
        n = min(args.batch, args.samples - trainer.n_training_samples)
        x = next(data) if n == args.batch else world.sample(n)
        out = trainer.step(x)
        trainer.n_training_steps += 1
        step = trainer.n_training_steps
        if r['resample_every'] and step % r['resample_every'] == 0 and frac < r['resample_until']:
            extra['resampled'] += resample_dead(sae, trainer, world, r, args.seed, step, args.device)
        if r['revive'] and r['arch'] == 'jr':
            with torch.no_grad():
                dead = trainer.n_forward_passes_since_fired > r['dead_window']
                top = out.hidden_pre.detach().float().amax(0)
                dead &= top > 0
                if dead.any():
                    sae.threshold.data[dead] = torch.minimum(sae.threshold.data[dead], 0.9 * top[dead])
                    extra['revived'] = extra.get('revived', 0) + int(dead.sum())
        if device_type == 'cuda':
            torch.cuda.synchronize(args.device)
        extra['train_seconds'] += time.monotonic() - start
        done = trainer.n_training_samples >= args.samples
        if step == 1 or step % interval == 0 or done:
            with torch.no_grad():
                acts = out.feature_acts
                l0 = float((acts != 0).sum(-1).float().mean())
                ev = 1.0 - float((out.sae_out.float() - out.sae_in.float()).square().sum(-1).mean()
                                 / (out.sae_in.float() - out.sae_in.float().mean(0)).square().sum(-1).mean())
                row = dict(samples=trainer.n_training_samples, steps=step, seconds=round(extra['train_seconds'], 1),
                           l0=l0, ev=ev, trainer_dead=int(trainer.dead_neurons.sum()),
                           losses={key: float(v) for key, v in out.losses.items()},
                           lr=trainer.optimizer.param_groups[0]['lr'], resampled=extra['resampled'],
                           revived=extra.get('revived', 0))
                if r['arch'] == 'jr':
                    row.update(bandwidth=sae.bandwidth, multiplier=sae.autotuner.multiplier,
                               smoothed_l0=sae.autotuner.smoothed_l0,
                               threshold_mean=float(sae.threshold.mean()), threshold_min=float(sae.threshold.min()))
                elif r['gate'] != 'plain':
                    row.update(gate_plain=sae.activation_fn.plain, gate_extra=sae.activation_fn.extra)
                if hasattr(sae, 'topk_threshold'):
                    row['topk_threshold'] = float(sae.topk_threshold)
            extra['history'].append(row)
            write_json(folder / 'progress.json', dict(status='trained' if done else 'training', **row))
            print(json.dumps(row), flush=True)
        planned_stop = args.stop_after_steps and step >= args.stop_after_steps
        if done or planned_stop or stop or time.monotonic() - last_save >= 60 * args.checkpoint_minutes:
            if r['gate'] != 'plain':
                extra['gate'] = gate_state(sae)
            save_checkpoint(checkpoint, trainer, extra)
            write_json(folder / 'history.json', extra['history'])
            last_save = time.monotonic()
        if stop or planned_stop:
            print('Checkpointed and stopping', stop, flush=True)
            return
    export_sae(sae, trainer, folder)
    if args.skip_eval:
        return
    print('EVALUATING', folder, flush=True)
    t0 = time.monotonic()
    metrics = score_saved(world, args, folder)
    result = dict(metadata, init=extra['init'], train_samples=trainer.n_training_samples,
                  train_seconds=extra['train_seconds'], eval_seconds=time.monotonic() - t0,
                  eval_samples=args.eval_samples, resampled=extra['resampled'],
                  final_trainer_dead=int(trainer.dead_neurons.sum()), **metrics)
    write_json(folder / 'score.json', result)
    checkpoint.unlink(missing_ok=True)
    print('RESULT', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
