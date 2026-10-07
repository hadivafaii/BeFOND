# BeFOND package

This package implements Bernoulli sparse coding with natural-gradient inference
and learning. Inputs are ordinary `[batch, input_dimension]` tensors. The model
has a factorized prior and a linear Gaussian decoder; it learns without
backpropagating through its inference trajectory.

Read the main components in this order:

| File or package | Purpose |
| --- | --- |
| [`model.py`](model.py) | `BeFOND`, `ModelConfig`, model parameters, inference results, and exact free energy |
| [`inference.py`](inference.py) | Stable Bernoulli moments, rolling prior references, and guarded ETD1, ETD1, and Euler inference |
| [`learning.py`](learning.py) | Decoder, bias, observation variance, and baseline-prior updates |
| [`training.py`](training.py) | Training loop, separate prior rollouts, validation, and logging |
| [`solvers.py`](solvers.py) | Dense and matrix-free decoder updates |
| [`distributed.py`](distributed.py) | Global-moment NGD across torchrun workers |
| [`checkpoints.py`](checkpoints.py) | Resume checkpoints, portable bundles, and research-fit conversion |
| [`data/`](data/) and [`evaluation/`](evaluation/) | Data preparation, normalization, and benchmark interfaces |

General defaults live in `ModelConfig` and [`TrainConfig`](config.py). The
[`experiment configurations`](../configs/) expose the current benchmark
settings. The default integrator is `etd1_guarded` for every dataset. See the
[algorithm guide](../docs/algorithm.md) for the equations and
[checkpoint guide](../docs/checkpoints.md) for loading trained models.

## Quick CPU checks

From the repository root, with Python 3.11 or newer:

```bash
pip install -e '.[test]'
python -m befond.training --config examples/toy.json --show-config
python -m pytest -q tests/test_math.py
```

The first command after installation prints resolved settings without training.
The tests check inference and learning against independent mathematical
references using small CPU tensors. Neither check downloads data or needs an
account. See [`tests/`](../tests/) for other small checks and the
[root README](../README.md) for a complete toy example.
