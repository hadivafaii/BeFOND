# Data streams and normalization

This package turns toy data, synthetic features, or cached language-model activations into tensors shaped `[batch, input_dimension]`. Training, prior fitting, and validation ask the same source for explicitly named streams.

For a small CPU check, install the base package with the `test` extra and run **from the repository root**:

```bash
python -m pytest tests/test_data.py -q
```

The three tests use generated tensors and temporary fixtures. They check keyed sampling, BF16 cache reads and shuffling, and normalization/decoder round trips; no dataset, model download, account, or GPU is needed.

| File | Read it for |
| --- | --- |
| [`__init__.py`](__init__.py) | `make_source(config, device)`, the lazy dataset factory. |
| [`toy.py`](toy.py) | A small binary sparse-coding generator with no external data. |
| [`synthetic.py`](synthetic.py) | Loading the public generator and reproducible batches keyed by seed, stream, and step. |
| [`gemma.py`](gemma.py) | Document-aware activation extraction and bounded training-buffer shuffling. |
| [`activation_cache.py`](activation_cache.py) | Verified BF16 activation shards and bounded prefetching. |
| [`normalization.py`](normalization.py) | Streaming moments, whitening/scaling, and conversion back to raw activation coordinates. |

A source's main interface is `sample(batch_size, step, stream="train")`. Synthetic sources also provide `sample_with_codes(...)` for evaluation against known features. [`synthetic_spec.py`](synthetic_spec.py) and [`gemma_spec.py`](gemma_spec.py) record pinned public data/model identities; editable experiment settings live in [`configs/`](../../configs/README.md).

Real synthetic data requires the optional `synthetic` environment. Extracting Gemma activations requires the separate optional `gemma` environment and model access; reading already-prepared shards uses the base dependencies. Preparation commands and storage requirements are documented under [`experiments/`](../../experiments/README.md).
