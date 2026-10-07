"""Two-worker global-moment NGD parity, with uneven sample and output shards."""

import datetime
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from befond.distributed import (infer_distributed, partition_batch,
                                update_parameters_distributed_, weighted_mean)
from befond.learning import update_parameters_
from befond.model import BeFOND, ModelConfig


def _compare_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        # Cover each update path and both parameter shapes without a cross-product.
        cases = [(False, "dense", "prior"),
                 (True, "matrix_free_phi1_auto", "agg_predictive_prior"),
                 (True, "capped_quadratic", "factorized_posterior")]
        for pixelwise_bias, solver, metric in cases:
            config = ModelConfig(input_dim=3, num_latents=4, seed=15,
                                 fit_dec_bias=True, dec_bias_pixelwise=pixelwise_bias,
                                 dec_var_pixelwise=pixelwise_bias, inference_h=1.5,
                                 inference_prior_retention=0.8, inference_prior_mix="prob")
            reference, distributed = BeFOND(config).double(), BeFOND(config).double()
            x = torch.tensor([[0.2, 0.5, -0.7], [0.4, -0.3, 0.1], [0.7, 0.5, -0.1],
                              [0.2, -0.6, 0.8], [0.5, 0.4, -0.7]], dtype=torch.float64)
            local_x = partition_batch(x)
            expected = reference.infer(x, steps=3, beta=0.8)
            local_result = infer_distributed(distributed, local_x, steps=3, beta=0.8)
            settings = dict(learning_rate=0.03, variance_rate=0.02, bias_rate=0.01,
                            beta=0.8, decoder_solver=solver, decoder_metric=metric,
                            fisher_damping=1e-6)
            expected_stats = update_parameters_(reference, x, expected, **settings)
            actual_stats = update_parameters_distributed_(distributed, local_x,
                                                           local_result, **settings)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(distributed.state_dict()[name], value,
                                           rtol=2e-8, atol=2e-10)
            assert abs(actual_stats["phi_delta_norm"] - expected_stats["phi_delta_norm"]) < 1e-9
            average = weighted_mean(local_x.mean(0), len(local_x))
            torch.testing.assert_close(average, x.mean(0))
    finally:
        dist.destroy_process_group()


def test_output_sharded_ngd_matches_single_worker(tmp_path):
    mp.spawn(_compare_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def _guarded_worker(rank, rendezvous):
    from befond.training import evaluate

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=30))
    try:
        model = BeFOND(ModelConfig(input_dim=1, num_latents=2,
                                   prior_init_lower=0.0, prior_init_upper=0.0,
                                   inference_integrator="etd1_guarded")).double()
        model.decoder.fill_(4.0)
        # The first shard accepts the full proposal; the last example needs
        # backtracking. Uneven shards must still use the global batch's gain.
        x = torch.tensor([[0.0], [0.5], [5.0]], dtype=torch.float64)
        local_x = partition_batch(x)
        local = model.infer(local_x, steps=1)
        local_gain = x.new_tensor(local.inference_gain)
        gains = [torch.empty_like(local_gain) for _ in range(2)]
        dist.all_gather(gains, local_gain)
        assert gains[0] > gains[1]

        for steps in (1, 3):
            expected = model.infer(x, steps=steps)
            actual = infer_distributed(model, local_x, steps=steps)
            assert actual.inference_gain == expected.inference_gain
            torch.testing.assert_close(actual.logits, partition_batch(expected.logits))
            torch.testing.assert_close(actual.mean, partition_batch(expected.mean))

        # Validation on rank zero must not enter inference collectives while
        # the other worker waits for it to finish.
        if rank == 0:
            source = SimpleNamespace(sample=lambda batch_size, *_args, **_kwargs: x[:batch_size])
            cfg = SimpleNamespace(eval_batch_size=3, eval_n_batches=1, kl_beta=1.0)
            metrics = evaluate(model, source, cfg, {"steps": 3})
            assert all(torch.isfinite(x.new_tensor(value)) for value in metrics.values())
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_guarded_inference_shares_global_gain_and_keeps_validation_local(tmp_path):
    mp.spawn(_guarded_worker, args=(str(tmp_path / "guarded-rendezvous"),),
             nprocs=2, join=True)
