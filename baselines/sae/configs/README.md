# Small SAE examples

[`toy.json`](toy.json) is a complete, short training example: TopK with 64 latents,
32 input dimensions, an activity budget of 4, and 200 training updates. Its data
are generated in memory; no dataset download is needed.

The `model`, `train`, and `data` sections select the architecture, optimizer
settings, and data source. Omitted model and optimizer fields use the documented
defaults in [`../config.py`](../config.py) and [`../training.py`](../training.py).

After installing the project (`python -m pip install -e .`), run **from the
repository root**:

```bash
python -m baselines.sae.train --config baselines/sae/configs/toy.json \
  --output outputs/sae-toy --device cpu
```

The command writes resolved settings, a training log, portable weights, and a
resume checkpoint. Choose a fresh output directory for a new run, or add
`--resume` to continue the same run. To compare architectures, copy the JSON and
change `model.model_type`; see the [package guide](../README.md) for model names.
