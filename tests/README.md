# Tests

These tests check the public BeFOND implementation with small CPU tensors and
temporary files. The ordinary suite needs no benchmark datasets, accounts, or
GPU. It includes a short toy-training continuation check and two-process
CPU/Gloo comparisons; it does not run a full experiment. Keep new tests focused
on numerical correctness, data integrity, and behavior that could invalidate a
training run or saved model. Prefer representative combinations over exhaustive
parameter products and avoid snapshots of recipe metadata.

## Run from the repository root

Use Python 3.11 or newer and install the test dependencies:

```bash
pip install -e '.[test]'
```

Start with the mathematical core, then run the ordinary suite:

```bash
python -m pytest -q tests/test_math.py
python -m pytest -q tests/test_inference.py
python -m pytest -q tests
```

| Test file | What it checks |
| --- | --- |
| [`test_math.py`](test_math.py) | Bernoulli natural gradients, each decoder solver and metric, prior constraints, and parameter updates |
| [`test_inference.py`](test_inference.py) | Guarded ETD1 against enumerated Bernoulli free energy, per-example acceptance, backtracking limits, and original integrators |
| [`test_data.py`](test_data.py) | Normalization, raw-coordinate decoding, cached activations, and deterministic data streams |
| [`test_training.py`](test_training.py) | Exact toy-training continuation, model bundles, checkpoint conversion, and historical integrator compatibility |
| [`test_evaluation.py`](test_evaluation.py) | Evaluation overrides change inference without changing learned weights |
| [`test_distributed.py`](test_distributed.py) | Two CPU workers reproduce single-worker NGD and guarded inference with uneven partitions; main-worker validation stays local |
| [`test_training_reference.py`](test_training_reference.py) | Optional comparison of complete dense and capped-quadratic training runs with the research trainer |

Run any listed file on its own with `python -m pytest -q tests/FILE.py`.
Reference comparisons skip unless `FONDV2_REFERENCE` points to a research
checkout with its dependencies available:

```bash
FONDV2_REFERENCE=/path/to/research python -m pytest -q tests/test_training_reference.py
```

Read [`befond/`](../befond/) for the implementation and
Comparison models have their own tests and dependency instructions under
[`baselines/`](../baselines/); they are separate from this suite.
