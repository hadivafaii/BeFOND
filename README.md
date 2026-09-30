# BeFOND

**Bernoulli free-energy online natural-gradient dynamics** for iterative sparse
coding. BeFOND infers a factorized Bernoulli posterior and learns a linear
Gaussian decoder with natural-gradient updates. It has no learned encoder and
does not backpropagate through the inference trajectory.

This repository also contains the available comparison implementations:
Miguel's mean-field and GAMP models, David's full-width synthetic SAE recipes,
and six PyTorch SAE families. Training, data preparation, evaluation, and
checkpoint loading live together here.

## Start with a small example

Python 3.11 or newer is required. Install PyTorch for your hardware, then:

```bash
git clone https://github.com/hadivafaii/BeFOND.git
cd BeFOND
pip install -e '.[test]'
python examples/train_toy.py --output outputs/toy
```

The example runs on CPU, generates its own small binary dataset, and needs no
account or downloads. It saves metrics, a resumable checkpoint, and a portable
model bundle. Check the resolved settings without starting a run:

```bash
python examples/train_toy.py --show-config
```

Use the model directly:

```python
import torch
from befond import BeFOND, ModelConfig

model = BeFOND(ModelConfig(input_dim=12, num_latents=24, fit_dec_var=False))
x = torch.randn(8, 12)
posterior = model.infer(x, steps=20, beta=1.0)
reconstruction = model.reconstruct(posterior.mean)
```

Observations are `[batch, input_dimension]`; decoder columns are features;
posterior means are `[batch, number_of_latents]`.

## Where to read the code

Each package has its own short guide with commands to inspect or test it:
[`befond`](befond/README.md), [`data`](befond/data/README.md),
[`evaluation`](befond/evaluation/README.md), [`baselines`](baselines/README.md),
and [`experiments`](experiments/README.md). For a first hands-on run, visit
[`examples`](examples/README.md); for checks, visit [`tests`](tests/README.md).
All guide commands run from the repository root unless stated otherwise.

[`docs/algorithm.md`](docs/algorithm.md) connects the equations, tensor shapes,
update order, and main hyperparameters.

1. [`model.py`](befond/model.py): model parameters, posterior moments, exact free energy.
2. [`inference.py`](befond/inference.py): Bernoulli inference and the rolling reference.
3. [`learning.py`](befond/learning.py): decoder, variance, bias, and initial-prior updates.
4. [`training.py`](befond/training.py): the complete training loop and separate prior rollouts.
5. [`solvers.py`](befond/solvers.py): matrix functions for larger dictionaries.

The small mathematical core and the benchmark-specific code are separate.
The numerical solvers preserve the research implementation's dense and
matrix-free update rules; they use portable PyTorch operations.

## Defaults and experiments

The configurations below reflect the **current development defaults at the
time of this release**, which are the intended starting point for the arXiv
version. They are not frozen recipes for the earlier conference manuscript.

| Experiment | Hyperparameters | Instructions |
| --- | --- | --- |
| Tiny CPU demonstration | [`examples/toy.json`](examples/toy.json) | Command above |
| SynthSAEBench | [`configs/synthetic.json`](configs/synthetic.json) | [`experiments/synthetic`](experiments/synthetic) |
| Synthetic superposition | [`configs/superposition.json`](configs/superposition.json) | [`experiments/superposition`](experiments/superposition) |
| Gemma-2-2B layer 12 | [`configs/gemma.json`](configs/gemma.json) | [`experiments/gemma`](experiments/gemma) |
| Comparison models | [`baselines/README.md`](baselines/README.md) | Separate model families and recipes |

General model defaults are defined in `ModelConfig` in [`model.py`](befond/model.py),
and training defaults in `TrainConfig` in [`config.py`](befond/config.py).
Each experiment JSON exposes model, training, data, and evaluation settings.
Every run saves the fully resolved configuration. Command-line overrides are
explicit, and misspelled keys fail:

```bash
python -m experiments.synthetic.train --show-config
python -m experiments.synthetic.train --device cuda:0 --output outputs/synthetic \
  --set model.num_latents=32768 --set train.lr=0.02
```

Changing width does not silently retune the prior probability or other
hyperparameters. Inspect and change those settings explicitly when needed.

To use multiple GPUs for one fit, launch the same configuration with torchrun:

```bash
torchrun --standalone --nproc_per_node=2 -m experiments.gemma.train \
  --device cuda --output outputs/gemma
```

Batch size remains global. Inference is partitioned across workers, global
posterior moments are shared, and the decoder solve is partitioned by output
coordinate. Baseline-prior rollouts are replicated; only the first worker writes
checkpoints, evaluates, and logs. CPU/Gloo execution supports small distributed
checks. Full-scale GPU throughput depends on width and solver conditioning.

Benchmark environments are optional. The synthetic, Gemma, and David pipelines
use different pinned SAE-Lens versions, so use separate environments as
described in their instructions. The core model and toy example do not require
any of those dependencies. Full benchmark preparation can require substantial
storage and compute; each preparation command documents its data budget.

For SynthSAEBench, create an environment and install its extra:

```bash
python3.11 -m venv .venv-synthetic
source .venv-synthetic/bin/activate
pip install -e '.[synthetic]'
```

For Gemma, use a separate Python 3.11 environment and `pip install -e '.[gemma]'`.
For Miguel and David, follow their package READMEs and pinned requirements;
install this repository into each environment with `pip install --no-deps -e .`.

## Checkpoints

A run writes `checkpoint.pt` for continuation and `pretrained/` for inference.
Bundles include model dimensions, all model parameters, metadata, and the
fitted normalization transform. They support arbitrary released widths.

```python
from befond.checkpoints import from_pretrained

model = from_pretrained("outputs/toy/pretrained", device="cpu")
# Once checkpoints are published:
# model = from_pretrained(repo_id, subfolder=width_folder, revision=commit, device="cuda:0")
```

The loader accepts an explicit Hugging Face repository, subfolder, and revision;
it does not assume a release already exists. Install `.[hub]` for Hub downloads.
See [`docs/checkpoints.md`](docs/checkpoints.md) for research-checkpoint conversion,
normalization, and the release layout. David's already-published synthetic SAE
checkpoints have their own loader in [`baselines/david`](baselines/david).

Continue a stopped run using its saved configuration:

```bash
python -m befond.training --config outputs/toy/config.json \
  --output outputs/toy --resume outputs/toy/checkpoint.pt
```

`train.stop_after_updates` can stop a run early while preserving its full
learning-rate schedule; set it back to `null` when continuing.

## Logging and validation

Local metrics are written to `metrics.jsonl`. W&B is optional (`pip install -e '.[logging]'`);
enable `logging.wandb` in the
configuration and provide your own entity/project. Its runtime files go to
`~/Projects/<project>/wandb/`.

```bash
pytest -q tests baselines/sae/tests
```

Tests compare the inference equations with enumerated Bernoulli free energy,
solver updates with independent matrix exponentials, and checkpoint/resume
behavior. Optional parity tests compare against the original research checkout
when `FONDV2_REFERENCE` points to it. Baseline-specific tests are documented in
their packages. See [`VALIDATION.md`](VALIDATION.md) for release verification.

## Scope and attribution

BeFOND contains Bernoulli NGD inference and learning. Other inference/learning
rules live in the comparison models. Poisson models, truncated-count models,
backpropagation-through-time training, research caches, and private paths are
excluded. David's Gemma-specific training source is pending; the supplied
synthetic recipes and this project's general SAE implementations are included.

The project is MIT licensed. Existing third-party notices are preserved; see
[`THIRD_PARTY.md`](THIRD_PARTY.md).
