# David's full-width synthetic SAE study

This package preserves the supplied portable training and inference code for
`synth_sae_bench/26_full_width_sweep`. It contains original and improved
BatchTopK, Matryoshka, and JumpReLU recipes selected for the paper's
original/modified comparison, with their exact per-run overrides.
No external private checkout is needed.

For a small CPU check, first use the dedicated SAELens environment described
under [Install](#install), with `pytest` installed. Run **from the repository root**:

```bash
python -m pytest baselines/david/tests -q
```

These checks use generated tensors and temporary checkpoints; they download no
world or checkpoint. They exercise the MinFire gate, training continuation,
resampling, and inference export.

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
  --k 40 --seed 0 --tag reproduction --out-root outputs/david --device cuda
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

The bundled [`catalog.json`](catalog.json) retains six selected seed-0 records
from the [published checkpoint release](https://huggingface.co/chanind-goodfire/synthsaebench-full-width-sweep):

| Method | Original target L0 | Modified target L0 |
| --- | ---: | ---: |
| BatchTopK | 35 | 45 |
| Matryoshka | 20 | 35 |
| JumpReLU | 45 | 40 |

These identify the source recipes for the paper's synthetic comparisons.
The catalog's scores are the published source evaluations, not the paper's
subsequent reevaluations or three-seed averages. The catalog contains no unused run records.

```python
from baselines.david.load import list_runs, load_sae

runs = list_runs()  # six selected paper recipes
sae = load_sae('grid/jr_ste_c01_resample_decay_k40_s0', device='cpu')
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
python -m baselines.david.experiment --output paper_plan.json
python -m baselines.david.experiment --run paper_plan.json --index 0 \
  --out-root outputs/david --device cuda
```

The plan contains only the six selected records and preserves their overrides.
Scheduling multiple plan rows is left to the user's execution environment.

The MinFire gate selects each eligible latent's strongest positive activation,
then fills the remaining batch-wide TopK budget. The improved recipe turns the
gate off for the last 10% of training so the inference threshold can recalibrate.

During the first 80% of training, resampling recipes periodically restart dead
latents from poorly reconstructed samples, reset the corresponding Adam moments,
and reset inactivity counters. JumpReLU's controller uses the preceding step's
multiplier, with allowed range 0.01–100, to adjust the L0 coefficient.

[`provenance.json`](provenance.json) records upstream source hashes and revisions.
[`LICENSE.autotuner`](LICENSE.autotuner) preserves the Decode Research MIT notice
for the coefficient controller. The supplied numerical implementation is retained;
the registry contains 81 named presets, while the catalog and execution plan
contain only the six selected paper runs.

```bash
pytest baselines/david/tests -q
```
