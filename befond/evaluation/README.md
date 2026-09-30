# Evaluation

This package measures synthetic feature recovery and adapts BeFOND's iterative inference to SAE Probes and RAVEL. It keeps evaluation readouts and raw activation coordinates separate from the training loop.

With the base package and `test` extra installed, run this small CPU check **from the repository root**:

```bash
python -m pytest tests/test_data.py::test_normalization_bundle_and_raw_decoder -q
```

It verifies normalization loading, decoder coordinates, input scaling, and decoder gradients using generated tensors. No benchmark packages, downloaded models, accounts, or GPU are required.

| File | What it does |
| --- | --- |
| [`synthetic.py`](synthetic.py) | Streams held-out examples and reports MCC, Macro F1, Rare F1, sparsity, and reconstruction error. |
| [`matching.py`](matching.py) | Matches learned dictionary atoms to known features using cosine similarity. |
| [`feature_metrics.py`](feature_metrics.py) | Scores feature detection and groups features by their independently estimated frequency. |
| [`adapter.py`](adapter.py) | Converts raw inputs to model coordinates, infers codes, and decodes into the original coordinates. |
| [`sae_probes.py`](sae_probes.py) | Calls upstream feature selection, probe fitting, and scoring. |
| [`ravel.py`](ravel.py) | Runs intervention-mask training and aggregates complete entity results. |

[`feature_cache.py`](feature_cache.py) bounds temporary feature storage; [`protocol.py`](protocol.py) supplies shared seeds and result metadata. The adapter freezes BeFOND while retaining decoder gradients needed by RAVEL's masks.

User-facing commands live under [`experiments/synthetic`](../../experiments/synthetic/README.md) and [`experiments/gemma`](../../experiments/gemma/README.md). Their `evaluate --help` commands work with the base installation. Actual synthetic evaluation needs the optional `synthetic` environment; SAE Probes/RAVEL need the separate optional `gemma` environment, checkpoint, and benchmark data described there.
