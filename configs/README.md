# Experiment configuration

This folder makes the current experiment hyperparameters visible in one place.
Edit a JSON file or override a setting on the command line; each run saves its
fully resolved configuration with the results.

Each experiment JSON groups its main settings into four parts:

- `model`: dictionary size, initialization, Bernoulli inference, learned prior/bias/variance.
- `train`: update counts, learning-rate schedule, inference-horizon sampling, NGD solvers, prior fits, and decoder constraints.
- `data`: benchmark identity, preparation paths, preprocessing, and training stream.
- `evaluation`: held-out sample counts, inference depth, and readout.

All runnable BeFOND configurations use `model.inference_integrator:
"etd1_guarded"`, including every Gemma width and seed. The Gemma configurations
are based on the selected paper models, with this updated integrator default.
Synthetic and superposition base files remain development presets; they are
not substitutes for the selected paper recipes. All synthetic inputs default
to unwhitened. See the [algorithm guide](../docs/algorithm.md#inference) for
guarded ETD1, the original `etd1`, and `euler`.

| Configuration | Input dimension | Dictionary width | Main updates | Preprocessing |
| --- | ---: | ---: | ---: | --- |
| [`synthetic.json`](synthetic.json) | 768 | 16,384 | 250,000 | None |
| [`gemma.json`](gemma.json) | 2,304 | 16,384 | 12,207 | Whitening fitted on 500M tokens |
| [`superposition.json`](superposition.json) | 768 | 4,096 | 8,000 of a 100,000-step schedule | None |

`gemma-{16,64,128,512}k-seed{0,1,2}.json` provides runnable recipes for all twelve
selected BeFOND models (widths 16,384, 65,536, 131,072, and 524,288).
`gemma.json` is the 16,384-latent seed-0 recipe. Final evaluation uses 10 steps;
training validation retains its recorded 100 steps. The 524,288-latent runs have
seed-specific settings, including a different evaluation reference retention
for seed 0. See the [paper model index](../experiments/gemma/PAPER_MODELS.md)
for exact configuration paths, baseline settings, and source run identities.

The selected synthetic seed-0 parameters are in
[`synthetic-paper-seed0.json`](synthetic-paper-seed0.json) (seven full-width
recovery methods) and [`superposition-paper-seed0.json`](superposition-paper-seed0.json)
(seven methods at each of six input dimensions, with 4,096 learned latents).
These parameter records retain native configuration names and identify the
figure, table, source run, evaluation settings, and seed-0 values. They contain
only plotted configurations, not search grids, and are reference records rather
than inputs to `befond-train`. The main-text figures aggregate three seeds;
these files document the seed-0 runs only. Their BeFOND entries retain the
original `etd1` integrator because these files record historical runs.

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

Unknown fields and invalid model/training values raise an error. Setting
`train_steps: null` resolves it from the token budget divided by the batch size.
Selected Gemma recipes explicitly record 12,207 updates.

Training overrides use `--set section.key=JSON`. For example, `--set model.num_latents=65536` changes dictionary width. Changing width alone does not retune the initialization, norm target, sparsity budget, or other settings. Keep every chosen setting with the resulting checkpoint.

To use the original unguarded ETD1 update from the recorded paper runs, pass
`--set model.inference_integrator='"etd1"'`. Existing checkpoints retain their
saved integrator; the new default applies to new models and runnable recipes.
