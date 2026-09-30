# Tests

These tests check the public BeFOND implementation with small CPU tensors and
temporary files. The ordinary suite needs no benchmark datasets, accounts, or
GPU. It includes a short toy-training continuation check and a two-process
CPU/Gloo comparison; it does not run a full experiment.

## Run from the repository root

Use Python 3.11 or newer and install the test dependencies:

```bash
pip install -e '.[test]'
```

Start with the mathematical core, then run the ordinary suite:

```bash
python -m pytest -q tests/test_math.py
python -m pytest -q tests
```

| Test file | What it checks |
| --- | --- |
| [`test_math.py`](test_math.py) | Bernoulli natural gradients, decoder solvers, prior constraints, and parameter updates |
| [`test_data.py`](test_data.py) | Normalization, raw-coordinate decoding, cached activations, and deterministic data streams |
| [`test_training.py`](test_training.py) | Exact toy-training continuation, model bundles, and checkpoint conversion |
| [`test_distributed.py`](test_distributed.py) | Two CPU workers reproduce single-worker NGD with uneven partitions |
| [`test_reference_parity.py`](test_reference_parity.py) | Optional comparison of trajectories and parameter updates with the research code |
| [`test_training_reference.py`](test_training_reference.py) | Optional comparison of complete short training runs with the research trainer |

Run any listed file on its own with `python -m pytest -q tests/FILE.py`.
Reference comparisons skip unless `FONDV2_REFERENCE` points to a research
checkout with its dependencies available:

```bash
FONDV2_REFERENCE=/path/to/research python -m pytest -q \
  tests/test_reference_parity.py tests/test_training_reference.py
```

Read [`befond/`](../befond/) for the implementation and
[`VALIDATION.md`](../VALIDATION.md) for the release checks and their limits.
Comparison models have their own tests and dependency instructions under
[`baselines/`](../baselines/); they are separate from this suite.
