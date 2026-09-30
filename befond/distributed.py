"""Multi-GPU NGD training with torchrun and replicated model parameters.

Inference splits a global minibatch across ranks. Decoder learning gathers the
posterior moment factors once, then divides the output rows of the decoder among
ranks. Each rank solves against the same global moments; averaging independent
local natural-gradient updates would give a different algorithm.
"""

import datetime
import os

import torch
import torch.distributed as dist

from .inference import bernoulli_moments, inference_direction
from .learning import exponential_relaxation, update_parameters_
from .solvers import decoder_update


def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def init_distributed(device="cpu"):
    """Initialize torchrun workers and select each worker's local CUDA device."""
    device = torch.device(device)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if device.type == "cuda":
        index = int(os.environ.get("LOCAL_RANK", "0")) if world_size > 1 else device.index
        device = torch.device("cuda", torch.cuda.current_device() if index is None else index)
        torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                timeout=datetime.timedelta(hours=6))
    return device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if is_distributed():
        dist.barrier()


def partition(size, rank, world_size):
    """Contiguous, order-preserving partition with at most one row of imbalance."""
    width, remainder = divmod(size, world_size)
    start = rank * width + min(rank, remainder)
    return slice(start, start + width + int(rank < remainder))


def partition_batch(global_batch):
    """Select this rank's examples from an identically sampled global batch."""
    if not is_distributed():
        return global_batch
    world = dist.get_world_size()
    if len(global_batch) < world:
        raise ValueError("Global batch size must be at least the number of workers.")
    return global_batch[partition(len(global_batch), dist.get_rank(), world)]


def gather_rows(local):
    """All-gather unequal first dimensions, retaining original sample order."""
    if not is_distributed():
        return local
    length = torch.tensor([len(local)], device=local.device, dtype=torch.int64)
    lengths = [torch.empty_like(length) for _ in range(dist.get_world_size())]
    dist.all_gather(lengths, length)
    lengths = [int(value.item()) for value in lengths]
    width = max(lengths)
    if width == 0:
        return local
    padded = local.new_zeros((width, *local.shape[1:]))
    padded[:len(local)].copy_(local)
    output = local.new_empty((width * len(lengths), *local.shape[1:]))
    dist.all_gather_into_tensor(output, padded.contiguous())
    return torch.cat([part[:length] for part, length in zip(output.split(width), lengths)])


def weighted_mean(value, count):
    """Reduce local means using their sample counts (including uneven batches)."""
    if not is_distributed():
        return value
    packed = torch.cat((value.reshape(-1) * count, value.new_tensor([count])))
    dist.all_reduce(packed)
    return (packed[:-1] / packed[-1]).reshape_as(value)


def _solver_diagnostics(diagnostics, device):
    # All ranks use the same operator, but convergence may need a different
    # number of terms for each output shard. Report the slowest shard's work.
    values = torch.tensor(list(diagnostics.values()), dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return {key: int(value) if isinstance(diagnostics[key], int) else value
            for key, value in zip(diagnostics, values.tolist())}


@torch.no_grad()
def update_parameters_distributed_(model, x, result, learning_rate, *, variance_rate=None,
                                   bias_rate=None, beta=1.0, delta_t=1.0,
                                   decoder_metric="prior", decoder_solver="dense",
                                   fisher_damping=0.0, chebyshev_degree=128,
                                   phi1_auto_taylor_max_applications=32,
                                   phi1_auto_chebyshev_tolerance=1e-5,
                                   phi1_auto_chebyshev_max_degree=512):
    """Update from local posteriors with global moments and output-row solves.

    Every rank must enter this function with identical model parameters. Bias
    and noise sufficient statistics are sample-weighted across ranks. The
    decoder, bias, and noise are replicated identically after the update.
    """
    arguments = dict(variance_rate=variance_rate, bias_rate=bias_rate, beta=beta,
                     delta_t=delta_t, decoder_metric=decoder_metric,
                     decoder_solver=decoder_solver, fisher_damping=fisher_damping,
                     chebyshev_degree=chebyshev_degree,
                     phi1_auto_taylor_max_applications=phi1_auto_taylor_max_applications,
                     phi1_auto_chebyshev_tolerance=phi1_auto_chebyshev_tolerance,
                     phi1_auto_chebyshev_max_degree=phi1_auto_chebyshev_max_degree)
    if not is_distributed():
        return update_parameters_(model, x, result, learning_rate, **arguments)
    world, rank = dist.get_world_size(), dist.get_rank()
    if model.config.input_dim < world:
        raise ValueError("Input dimension must be at least the number of workers.")
    variance_rate = learning_rate if variance_rate is None else variance_rate
    bias_rate = learning_rate if bias_rate is None else bias_rate
    phi = model.decoder.detach()
    bias = model.bias.detach() if model.bias is not None else None
    centered = x if bias is None else x - bias
    variance_target = ((centered - result.mean @ phi.T).square()
                       + result.variance @ phi.square().T).mean(0)
    variance_target = weighted_mean(variance_target, len(x))
    bias_new = None
    if bias is not None:
        if result.inner_steps != 1:
            raise ValueError("Corrected bias learning requires one inner inference step.")
        direction = inference_direction(model, x, result.logits, result.reference_logits, beta)
        response = result.mean + result.inference_gain * result.variance * direction
        residual = x - response @ phi.T
        if bias.numel() == 1:
            precision = model.variance.reciprocal().expand_as(residual)
            target = (precision * residual).sum() / precision.sum()
        else:
            target = residual.mean(0)
        target = weighted_mean(target, len(x))
        bias_new = exponential_relaxation(bias, target, bias_rate, delta_t)
    new_variance = None
    if model.config.fit_dec_var and variance_rate != 0:
        target = variance_target if model.log_variance.numel() > 1 else variance_target.mean()
        new_variance = exponential_relaxation(model.variance, target, variance_rate, delta_t)
        if not torch.isfinite(new_variance).all() or (new_variance <= 0).any():
            raise ValueError("Observation variance must stay finite and positive.")

    global_x = gather_rows(centered)
    mean, variance = gather_rows(result.mean), gather_rows(result.variance)
    if decoder_metric in ("prior", "agg_posterior", "factorized_posterior"):
        prior_mean, prior_variance = bernoulli_moments(model.prior_logits.unsqueeze(0))
    else:
        prior_mean = gather_rows(result.reference_mean)
        prior_variance = gather_rows(result.reference_variance)
    rows = partition(len(phi), rank, world)
    precision = model.variance.reciprocal().expand(len(phi))[rows]
    decoder_new, diagnostics = decoder_update(
        phi=phi[rows], x=global_x[:, rows], mean=mean, variance=variance,
        prior_mean=prior_mean, prior_variance=prior_variance,
        learning_rate=learning_rate, delta_t=delta_t, decoder_metric=decoder_metric,
        decoder_solver=decoder_solver, fisher_damping=fisher_damping,
        chebyshev_degree=chebyshev_degree,
        phi1_auto_taylor_max_applications=phi1_auto_taylor_max_applications,
        phi1_auto_chebyshev_tolerance=phi1_auto_chebyshev_tolerance,
        phi1_auto_chebyshev_max_degree=phi1_auto_chebyshev_max_degree,
        observation_precision=precision, _reduce_line_statistics=dist.all_reduce)
    decoder_new = gather_rows(decoder_new)
    diagnostics = _solver_diagnostics(diagnostics, x.device)
    diagnostics.update(phi_delta_norm=(decoder_new - phi).norm().item(),
                       bias_delta_norm=(0.0 if bias_new is None else (bias_new - bias).norm().item()),
                       var_target_mean=variance_target.mean().item(),
                       var_target_max=variance_target.max().item(), batch_size=len(global_x))
    model.decoder.copy_(decoder_new)
    if bias_new is not None:
        model.bias.copy_(bias_new)
    if new_variance is not None:
        model.log_variance.copy_(new_variance.log())
    return diagnostics
