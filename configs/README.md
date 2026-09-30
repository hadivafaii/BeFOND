# Experiment configuration

This folder makes the current experiment hyperparameters visible in one place.
Edit a JSON file or override a setting on the command line; each run saves its
fully resolved configuration with the results.

Each experiment JSON groups its main settings into four parts:

- `model`: dictionary size, initialization, Bernoulli inference, learned prior/bias/variance.
- `train`: update counts, learning-rate schedule, inference-horizon sampling, NGD solvers, prior fits, and decoder constraints.
- `data`: benchmark identity, preparation paths, preprocessing, and training stream.
- `evaluation`: held-out sample counts, inference depth, and readout.

The checked-in values come from the current development-code defaults. They are intended to remain understandable and editable independently of older manuscript settings.

| Configuration | Input dimension | Dictionary width | Main updates | Preprocessing |
| --- | ---: | ---: | ---: | --- |
| [`synthetic.json`](synthetic.json) | 768 | 16,384 | 250,000 | Whitening |
| [`gemma.json`](gemma.json) | 2,304 | 16,384 | 12,207 | Whitening |
| [`superposition.json`](superposition.json) | 768 | 4,096 | 8,000 of a 100,000-step schedule | None |

[`superposition_sweep.json`](superposition_sweep.json) describes a grid;
`experiments.superposition.sweep` expands it into runnable configurations.
Preparation and evaluation commands are documented under
[`experiments/`](../experiments/README.md).

## Inspect and check a configuration

From the repository root, print the resolved settings without training or
downloading data. These commands only need the core installation:

```bash
pip install -e .
python -m experiments.synthetic.train --show-config
python -m experiments.gemma.train --show-config
python -m experiments.superposition.train --show-config
```

Try an override and inspect the resulting settings:

```bash
python -m experiments.synthetic.train --show-config \
  --set model.num_latents=32768 --set train.lr=0.02
```

Unknown fields and invalid model/training values raise an error. Gemma's
`train_steps: null` resolves from the token budget divided by the batch size
(12,207 updates with the checked-in defaults).

Training overrides use `--set section.key=JSON`. For example, `--set model.num_latents=65536` changes dictionary width. Changing width alone does not retune the initialization, norm target, sparsity budget, or other settings. Keep every chosen setting with the resulting checkpoint.
