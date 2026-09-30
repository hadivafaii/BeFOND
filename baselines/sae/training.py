"""Adam training for the small SAE implementations.

The loss, decoder-gradient projection, decoder normalization, dead-feature
accounting and warmup/cosine schedule follow the research implementation.
Data sources use the same keyed batches as BeFOND experiments.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch

from .checkpoints import save_sae


@dataclass
class TrainingConfig:
    train_steps: int = 12_000
    batch_size: int = 200
    lr: float = 1e-3
    lr_min: float = 1e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    grad_clip: float | None = None
    dead_feature_window: int = 1_000_000
    log_freq: int = 10
    checkpoint_freq: int = 1000


def lr_multiplier(step, cfg):
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return max(1, step + 1) / cfg.warmup_steps
    remaining = max(1, cfg.train_steps - cfg.warmup_steps)
    progress = min(1., max(0., (step - cfg.warmup_steps) / remaining))
    minimum = cfg.lr_min / cfg.lr
    return minimum + (1. - minimum) * .5 * (1. + math.cos(math.pi * progress))


def train(model, source, config=None, *, output, resume=False):
    """Train on ``source.sample(batch, step)``; save weights and resume state.

    ``resume=True`` requires the same model, training, and data configuration.
    Keyed data sampling restarts at the next unconsumed step.
    """
    import json

    cfg = config or TrainingConfig()
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    device = model.W_dec.device
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, cfg))
    tokens_since_fired = torch.zeros(model.num_latents, dtype=torch.long, device=device)
    step = 0
    checkpoint = path / 'training.pt'
    if resume:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state['model_config'] != asdict(model.config) or state['train_config'] != asdict(cfg):
            raise ValueError('Resume requires matching model and training configuration')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        tokens_since_fired.copy_(state['tokens_since_fired'])
        step = state['step']
    elif checkpoint.exists():
        raise FileExistsError(f'{checkpoint}: use resume=True or a new output directory')
    with (path / 'history.jsonl').open('a') as history:
        while step < cfg.train_steps:
            model.train()
            x = source.sample(cfg.batch_size, step).to(device=device, dtype=torch.float32)
            dead = tokens_since_fired >= cfg.dead_feature_window
            optimizer.zero_grad(set_to_none=True)
            out = model(x, dead_mask=dead)
            losses = model.compute_loss(x, output=out, dead_mask=dead)
            losses['loss'].backward()
            model.prepare_optimizer_step()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            model.post_optimizer_step()
            scheduler.step()
            active = out.latents.detach().ne(0)
            tokens_since_fired += len(x)
            tokens_since_fired[active.any(0)] = 0
            step += 1
            if step % cfg.log_freq == 0 or step == cfg.train_steps:
                row = dict(step=step, **{name: float(value.detach()) for name, value in losses.items()},
                           l0=float(active.sum(1).float().mean()), lr=optimizer.param_groups[0]['lr'])
                history.write(json.dumps(row) + '\n')
                history.flush()
                print(json.dumps(row), flush=True)
            if step % cfg.checkpoint_freq == 0 or step == cfg.train_steps:
                temporary = checkpoint.with_suffix('.tmp')
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(), tokens_since_fired=tokens_since_fired,
                                step=step, model_config=asdict(model.config), train_config=asdict(cfg)), temporary)
                temporary.replace(checkpoint)
                save_sae(model, path)
    return model.eval()
