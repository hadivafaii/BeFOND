"""NGD learning: infer a minibatch, update the model once, then fit the prior.

Training never differentiates through inference. The baseline prior has its own
data stream and longer rollouts; it is distinct from the rolling reference used
inside each input's inference trajectory.
"""

import argparse
import json
import math
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint, save_checkpoint, save_pretrained
from .config import TrainConfig, json_ready, load_config, resolve_config, write_config
from .horizon import horizon_probabilities
from .inference import bernoulli_kl
from .learning import project_prior_mean, update_prior_
from .distributed import (barrier, cleanup_distributed, init_distributed,
                          is_main_process, partition_batch,
                          update_parameters_distributed_, weighted_mean)
from .model import BeFOND, ModelConfig
from .schedules import LRSchedule


@torch.no_grad()
def regularize_decoder_(model, cfg, step, step_size):
    """Integrate the original radial norm penalty after the NGD update."""
    if cfg.decoder_norm_method == "none":
        return
    norms = model.decoder.norm(dim=0)
    safe = norms.clamp_min(1e-12)
    target = cfg.decoder_norm_target
    if cfg.decoder_norm_method == "clip":
        lower, upper = target if isinstance(target, (list, tuple)) else (target, target)
        factor = safe.clamp(lower, upper) / safe
    else:
        anneal_steps = math.ceil(cfg.train_steps * cfg.decoder_norm_lamb_anneal_portion)
        fraction = min(step / max(anneal_steps - 1, 1), 1) if anneal_steps else 1
        strength = cfg.decoder_norm_lamb_start + fraction * (
            cfg.decoder_norm_lamb_end - cfg.decoder_norm_lamb_start)
        exponent = step_size * strength
        deviation = (norms - target).abs()
        radius = 2.0
        quartic_time = (0.5 * (1 - (radius / deviation.clamp_min(radius)).square())).clamp_max(exponent)
        log_retention = (-0.5 * torch.log1p(2 * quartic_time * (deviation / radius).square())
                         - (exponent - quartic_time))
        factor = log_retention.exp() - log_retention.expm1() * target / safe
    model.decoder.mul_(factor.unsqueeze(0))


@torch.no_grad()
def prior_target(model, source, cfg, step):
    """Infer the baseline-prior target; reject unchecked or unstable rollouts."""
    auto = cfg.ngd_prior_t_outer == "auto"
    maximum = cfg.ngd_prior_t_outer_max if auto else int(cfg.ngd_prior_t_outer)
    threshold = cfg.ngd_prior_kl_threshold
    if isinstance(threshold, (tuple, list)):
        stage = min(step * len(threshold) // cfg.train_steps, len(threshold) - 1)
        threshold = threshold[stage]
    window_size = cfg.ngd_prior_kl_window
    event = step // cfg.ngd_prior_update_every
    rate_sum, count, rejected, depths = None, 0, False, []
    for index in range(cfg.ngd_prior_n_batches):
        x = source.sample(cfg.ngd_prior_batch_size or cfg.batch_size,
                          event * cfg.ngd_prior_n_batches + index, stream="prior")
        logits = model.initial_logits(len(x))
        drive, diagonal = model.inference_terms(x)
        window, passing = deque(maxlen=window_size), 0
        for depth in range(1, maximum + 1):
            previous = logits
            result = model.inference_step(x, previous, beta=cfg.kl_beta,
                                           drive=drive, gram_diagonal=diagonal)
            logits = result.logits
            if auto and depth > window_size:
                movement = bernoulli_kl(logits, previous).sum(-1).mean().item()
                window.append(movement)
                if len(window) == window_size:
                    passing = passing + 1 if sum(window) / window_size <= threshold else 0
                    if passing >= 3:
                        break
        if auto:
            checked = len(window) == window_size
            final = window[-1] if checked else math.nan
            average = sum(window) / window_size if checked else math.nan
            rejected |= any(not math.isfinite(value) or
                            value / model.config.num_latents >= cfg.ngd_prior_kl_reject_threshold
                            for value in (final, average))
        current = result.mean.sum(0)
        rate_sum = current if rate_sum is None else rate_sum + current
        count += len(x)
        depths.append(depth)
    return rate_sum / count, {"prior_rejected": int(rejected),
                              "prior_inference_steps": sum(depths) / len(depths),
                              "prior_examples": count}


@torch.no_grad()
def evaluate(model, source, cfg, evaluation, sample_count=None):
    totals = {}
    count = 0
    steps = evaluation.get("steps", model.config.t_outer)
    if sample_count is None:
        steps = evaluation.get("validation_steps", steps)
    sample_count = sample_count or cfg.eval_batch_size * cfg.eval_n_batches
    for index, offset in enumerate(range(0, sample_count, cfg.eval_batch_size)):
        x = source.sample(min(cfg.eval_batch_size, sample_count - offset), index, stream="validation")
        result = model.infer(x, steps=steps, beta=cfg.kl_beta)
        metrics = model.loss(x, result, cfg.kl_beta)
        metrics["l0"] = (result.mean > 0.5).sum(-1).to(x.dtype)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value.sum().item()
        count += len(x)
    return {f"eval/{name}": value / count for name, value in totals.items()}


def wandb_project_dir(project):
    """Each project owns its runtime logs outside the checkout and checkpoints."""
    return Path.home() / "Projects" / project / "wandb"


def _open_wandb(config):
    options = config.get("logging", {})
    if not options.get("wandb", False):
        return None
    import wandb
    project = options.get("project", "BeFOND")
    directory = wandb_project_dir(project)
    directory.mkdir(parents=True, exist_ok=True)
    return wandb.init(project=project, entity=options.get("entity"), dir=str(directory),
                      config=json_ready(config), name=options.get("name"))


@torch.no_grad()
def train(config, output_dir, device="cpu", resume=None, *, source=None):
    """Run one experiment and return the trained model.

    Data sources expose sample(batch_size, step, stream), making training/prior/
    validation streams explicit. An optional source state is saved on resume.
    ``stop_after_updates`` stops at an absolute update count without shortening
    the learning-rate schedule, so staged runs match uninterrupted training.
    """
    config = resolve_config(config)
    cfg = TrainConfig(**config["train"])
    device = init_distributed(device)
    main_process = is_main_process()
    torch.manual_seed(config["model"]["seed"])
    model = BeFOND(ModelConfig(**config["model"])).to(device)
    start = 0
    state = {}
    if resume is not None:
        model, state = load_checkpoint(resume, device)
        for section in ("model", "train", "data"):
            expected, actual = dict(json_ready(config[section])), dict(state["config"][section])
            if section == "train":
                for key in ("stop_after_updates", "eval_freq", "chkpt_freq", "log_freq"):
                    expected.pop(key, None)
                    actual.pop(key, None)
            if expected != actual:
                raise ValueError(f"Resume {section} configuration differs from the checkpoint")
        start = state["step"]
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in state:
            torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    if source is None:
        from .data import make_source
        source = make_source(config, device=device)
    if state.get("source_state") is not None:
        source.load_state_dict(state["source_state"])
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume and (output_dir / "checkpoint.pt").exists():
        raise FileExistsError("Output contains a checkpoint; use --resume or a new output directory")
    if main_process:
        write_config(output_dir / "config.json", config)
    probabilities = horizon_probabilities(model.config.t_outer, cfg.horizon_dist)
    schedule = LRSchedule(cfg.scheduler_type, cfg.lr, cfg.lr_min, cfg.train_steps,
                          cfg.warmup_portion, cfg.warm_restart, cfg.warm_restart_peak_mult)
    if start == 0:
        initial_norm = cfg.initial_decoder_norm
        if initial_norm is None and cfg.decoder_norm_method == "soft" and model.config.init_scale is None:
            initial_norm = cfg.decoder_norm_target
        if initial_norm is not None:
            model.decoder.mul_(initial_norm / model.decoder.norm(dim=0).clamp_min(1e-12))
        model.prior_logits.copy_(torch.logit(project_prior_mean(
            model.prior, cfg.r0_budget, cfg.r0_floor)))
    run = _open_wandb(config) if main_process else None
    stop = min(cfg.train_steps, cfg.stop_after_updates or cfg.train_steps)
    completed = start
    try:
        with ((output_dir / "metrics.jsonl").open("a") if main_process else nullcontext()) as log:
            for step in range(start, stop):
                x = partition_batch(source.sample(cfg.batch_size, step, stream="train"))
                rng = np.random.default_rng([model.config.seed, step])
                depth = int(rng.choice(model.config.t_outer, p=probabilities)) + 1
                rate = schedule(step)
                result = model.infer(x, steps=depth, beta=cfg.kl_beta)
                losses = model.loss(x, result, cfg.kl_beta)
                averages = weighted_mean(torch.stack([value.mean() for value in losses.values()]), len(x))
                metrics = dict(zip(losses, averages.tolist()))
                frozen = (cfg.ngd_freeze_variance_after is not None and
                          step >= cfg.ngd_freeze_variance_after and
                          (cfg.ngd_freeze_variance_scope == "all" or depth > 1))
                update = update_parameters_distributed_(
                    model, x, result, rate, beta=cfg.kl_beta,
                    variance_rate=0.0 if frozen else rate * cfg.ngd_lr_sigma_mult,
                    bias_rate=rate * cfg.ngd_lr_bias_mult, delta_t=cfg.ngd_delta_t,
                    decoder_metric=cfg.ngd_decoder_metric, decoder_solver=cfg.ngd_decoder_solver,
                    fisher_damping=cfg.ngd_fisher_damping,
                    chebyshev_degree=cfg.ngd_chebyshev_degree,
                    phi1_auto_taylor_max_applications=cfg.ngd_phi1_auto_taylor_max_applications,
                    phi1_auto_chebyshev_tolerance=cfg.ngd_phi1_auto_chebyshev_tolerance,
                    phi1_auto_chebyshev_max_degree=cfg.ngd_phi1_auto_chebyshev_max_degree)
                regularize_decoder_(model, cfg, step, rate * cfg.ngd_delta_t)
                if (model.config.fit_prior and rate * cfg.ngd_lr_prior_mult > 0 and
                        cfg.kl_beta > 0 and (step + 1) % cfg.ngd_prior_update_every == 0):
                    target, prior_stats = prior_target(model, source, cfg, step)
                    if not prior_stats["prior_rejected"]:
                        prior_stats.update(update_prior_(model, target,
                            rate * cfg.ngd_lr_prior_mult, cfg.kl_beta, cfg.ngd_delta_t,
                            budget=cfg.r0_budget, floor=cfg.r0_floor))
                    metrics.update(prior_stats)
                completed = step + 1
                metrics.update(step=completed, lr=rate, inference_steps=depth)
                metrics.update({key: float(value) for key, value in update.items()
                                if isinstance(value, (int, float)) or torch.is_tensor(value) and value.numel() == 1})
                if main_process and cfg.eval_freq and completed % cfg.eval_freq == 0:
                    metrics.update(evaluate(model, source, cfg, config.get("evaluation", {})))
                if main_process and (completed % cfg.log_freq == 0 or completed == stop):
                    log.write(json.dumps(metrics) + "\n")
                    log.flush()
                    print(f"step={completed} free_energy={metrics['free_energy']:.5g} lr={rate:.4g}", flush=True)
                    if run is not None:
                        run.log(metrics, step=completed)
                if main_process and cfg.chkpt_freq and completed % cfg.chkpt_freq == 0:
                    save_checkpoint(model, output_dir / "checkpoint.pt", config=config,
                        step=completed, source_state=source.state_dict() if hasattr(source, "state_dict") else None)
                barrier()
        if main_process:
            if completed == cfg.train_steps and cfg.eval_final_n_samples is not None:
                final_metrics = evaluate(model, source, cfg, config.get("evaluation", {}),
                                         sample_count=cfg.eval_final_n_samples)
                write_config(output_dir / "final_metrics.json", final_metrics)
            save_checkpoint(model, output_dir / "checkpoint.pt", config=config, step=completed,
                            source_state=source.state_dict() if hasattr(source, "state_dict") else None)
            normalization = None
            if hasattr(source, "normalizer"):
                normalization = output_dir / "normalization.pt"
                torch.save(source.normalizer.state_dict(), normalization)
            save_pretrained(model, output_dir / "pretrained", normalization_path=normalization,
                             metadata={"step": completed, "config": config})
        barrier()
    finally:
        if run is not None:
            run.finish()
    return model


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, required=default_config is None)
    parser.add_argument("--output", default="outputs/befond")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--show-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.show_config:
        print(json.dumps(json_ready(config), indent=2, allow_nan=False))
        return
    try:
        train(config, args.output, args.device, args.resume)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
