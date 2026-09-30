"""Two-worker global-moment NGD parity, with uneven sample and output shards."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from befond.distributed import partition_batch, update_parameters_distributed_, weighted_mean
from befond.learning import update_parameters_
from befond.model import BeFOND, ModelConfig


def _compare_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        cases = [("dense", "prior"), ("matrix_free_phi1_auto", "agg_predictive_prior"),
                 ("capped_quadratic", "factorized_posterior")]
        for pixelwise_bias in (False, True):
            for solver, metric in cases:
                config = ModelConfig(input_dim=3, num_latents=4, seed=15,
                                     fit_dec_bias=True, dec_bias_pixelwise=pixelwise_bias,
                                     dec_var_pixelwise=pixelwise_bias, inference_h=1.5,
                                     inference_prior_retention=0.8, inference_prior_mix="prob")
                reference, distributed = BeFOND(config).double(), BeFOND(config).double()
                x = torch.tensor([[0.2, 0.5, -0.7], [0.4, -0.3, 0.1], [0.7, 0.5, -0.1],
                                  [0.2, -0.6, 0.8], [0.5, 0.4, -0.7]], dtype=torch.float64)
                local_x = partition_batch(x)
                expected = reference.infer(x, steps=3, beta=0.8)
                local_result = distributed.infer(local_x, steps=3, beta=0.8)
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
