# SAE architectures

Each file implements an encoder, linear decoder, and training loss. All models
return `SAEOutput` and share the `encode`, `decode`, and `compute_loss` interface.
Read [`base.py`](base.py) for that interface, then one architecture file.

| File | What it contains |
| --- | --- |
| [`relu.py`](relu.py) | ReLU activations with an L1 sparsity penalty |
| [`gated.py`](gated.py) | Separate feature-selection and magnitude paths |
| [`topk.py`](topk.py) | TopK and BatchTopK, including dead-feature reconstruction |
| [`jumprelu.py`](jumprelu.py) | Learned activation thresholds and their gradient estimator |
| [`matryoshka.py`](matryoshka.py) | BatchTopK trained at nested dictionary widths |
| [`__init__.py`](__init__.py) | The `build_sae(config)` model factory |

Weights use one decoder row per feature: `[num_latents, input_dim]`.
Architecture settings live in [`../config.py`](../config.py), and the training
loop is in [`../training.py`](../training.py).

With the project and test extra installed (`python -m pip install -e ".[test]"`),
run this small check **from the repository root**:

```bash
python -m pytest baselines/sae/tests/test_models.py -k training_and_checkpoint_roundtrip -q
```

It trains one update for each architecture and verifies that saving and reloading
preserves its encoded activations. It runs on CPU without dataset downloads.
