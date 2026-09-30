# Experiment configuration

Each runnable JSON has four parts:

- `model`: dictionary size, initialization, Bernoulli inference, learned prior/bias/variance.
- `train`: update counts, learning-rate schedule, inference-horizon sampling, NGD solvers, prior fits, and decoder constraints.
- `data`: benchmark identity, preparation paths, preprocessing, and training stream.
- `evaluation`: held-out sample counts, inference depth, and readout.

The checked-in values come from the current development-code defaults. They are intended to remain understandable and editable independently of older manuscript settings.

| Configuration | Input dimension | Dictionary width | Main updates | Preprocessing |
| --- | ---: | ---: | ---: | --- |
| `synthetic.json` | 768 | 16,384 | 250,000 | Whitening |
| `gemma.json` | 2,304 | 16,384 | 12,207 | Whitening |
| `superposition.json` | 768 | 4,096 | 8,000 of a 100,000-step schedule | None |

`superposition_sweep.json` describes a grid; `experiments.superposition.sweep` expands it into runnable configurations. Preparation and evaluation commands are documented beside each experiment.

Training overrides use `--set section.key=JSON`. For example, `--set model.num_latents=65536` changes dictionary width. Changing width alone does not retune the initialization, norm target, sparsity budget, or other settings. Keep every chosen setting with the resulting checkpoint.
