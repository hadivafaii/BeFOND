# David's full-width synthetic SAE study

This package preserves the supplied portable training and inference code for
`synth_sae_bench/26_full_width_sweep`. It contains original and improved
BatchTopK, Matryoshka, and JumpReLU recipes, including the full screening recipe
registry and exact overrides for the selected L0-grid and MinFire follow-ups.
No external private checkout is needed.

Start with [`recipes.py`](recipes.py), then [`sae.py`](sae.py), then
[`train.py`](train.py). Training uses native SAELens losses and optimizer steps;
[`gates.py`](gates.py) contains the custom MinFire and Sinkhorn gates.

## Install

Use a separate environment from BeFOND's synthetic generator environment because
this study pins SAELens 6.51.0 at source revision
`b5711e34d072846dc112881f4c8d209407a32b56`:

```bash
python3 -m venv .venv-david
source .venv-david/bin/activate
pip install -r baselines/david/requirements.txt
pip install --no-deps -e .
```

The requirements preserve the supplied study environment. They include the
SAELens generator, scorer, native AuxK losses, JumpReLU straight-through estimator
and pre-activation loss. The package was originally validated with Python 3.12.

## Hyperparameters and training

| Architecture | Original recipe | Improved recipe |
| --- | --- | --- |
| BatchTopK | `btk` | `btk_minfire_decay` |
| Matryoshka | `mat` | `mat_resample_win100_decay` |
| JumpReLU | `jr` | `jr_ste_c01_resample_decay` |

Every recipe override is relative to the documented `BASE` dictionary in
[`recipes.py`](recipes.py). Common training defaults are 200,000,000 samples,
batch 1,024, 16,384 latents, 768 inputs, Adam (0.9, 0.999), learning rate 3e-4,
and BF16 autocast. Matryoshka prefixes are 128/512/2,048/16,384. JumpReLU estimates
input scaling from 1,000 batches. The improved recipes enable final-20% LR decay;
other changes are explicit in the registry.

```bash
python -m baselines.david.train --recipe jr_ste_c01_resample_decay \
  --k 35 --seed 0 --tag reproduction --out-root outputs/david --device cuda
```

Use `--override '{"minfire_every": 5}' --suffix _p5` for an explicit recipe
variant. `--samples`, `--batch`, `--width`, `--lr`, and `--eval-samples` override
the common training defaults. For a short local check, reduce width/sample count,
pass `--device cpu --skip-eval`, and disable autocast with
`--override '{"autocast": false}'`.

The world downloads from `decoderesearch/synth-sae-bench-16k-v1` at revision
`b2efd8b919ae46d6d487c73d46db5ee52813621d`, with parent scaling disabled. Use
`--offline` when cached or `--world-path` for a local copy. Saved models are scored
using the unchanged SAELens scorer on 1,000,000 fresh samples at seed `49100 + seed`.
MCC is one-to-one Hungarian matching over the full dictionary; F1 is the official
macro average. Full-width MCC is not directly comparable with a narrower board.

Each run writes resolved `run.json`, progress/history/score JSON, and
`final/{cfg.json,sae_weights.safetensors}`. Atomic `latest.pt` files preserve
optimizer/scheduler state, trainer counters, autotuner buffers, gate state and
RNG state. Rerunning a command resumes. SIGTERM/SIGUSR1 saves at the next step;
`--stop-after-steps` provides a planned stop while preserving the full horizon.
Resume files are local PyTorch serialization; public inference weights use
safetensors. Successful final scoring removes the local resume file.

## Published inference models

The existing [synthetic checkpoint release](https://huggingface.co/chanind-goodfire/synthsaebench-full-width-sweep)
contains 209 full-budget runs. The bundled [`catalog.json`](catalog.json) is
retained for selective loading and exact experiment plans; training logs and raw
research outputs are not copied into this repository.

```python
from baselines.david.load import list_runs, load_sae

runs = list_runs(headline_only=True)  # 42 models for the seven-point curves
sae = load_sae('r3/jr_ste_c01_resample_decay_k35_s0', device='cpu')
codes = sae.encode(x)
reconstruction = sae.decode(codes)
```

The loader downloads only the selected model's config and weights and pins Hub
revision `13630f6252a41d01e645f5c6714d39cafdb65f00`. Pass `revision=` to select
another revision or `local_files_only=True` to work offline. All three training
architectures export ordinary SAELens JumpReLU inference models. Their learned
thresholds and activation normalization are folded into the saved model: provide
unnormalized synthetic observations and do not normalize again.

## Experiment plans

```bash
python -m baselines.david.experiment --suite headline --output headline.json
python -m baselines.david.experiment --run headline.json --index 0 \
  --out-root outputs/david --device cuda
```

`--suite screen` contains 162 runs; `--suite all` contains all 209 released runs.
Plans preserve the original per-run overrides, including low-L0 MinFire periods.
Scheduling multiple plan rows is left to the user's execution environment.

The MinFire gate selects each eligible latent's strongest positive activation,
then fills the remaining batch-wide TopK budget. The improved recipe turns the
gate off for the last 10% of training so the inference threshold can recalibrate.
The original two-fire and Sinkhorn screening variants can exceed the TopK budget;
they are retained with their original behavior.

During the first 80% of training, resampling recipes periodically restart dead
latents from poorly reconstructed samples, reset the corresponding Adam moments,
and reset inactivity counters. JumpReLU's controller uses the preceding step's
multiplier, with allowed range 0.01–100, to adjust the L0 coefficient.

[`provenance.json`](provenance.json) records upstream source hashes and revisions.
[`LICENSE.autotuner`](LICENSE.autotuner) preserves the Decode Research MIT notice
for the coefficient controller. The supplied numerical implementation is retained;
only import paths, output defaults and environment-specific instructions changed.

```bash
pytest baselines/david/tests -q
```
