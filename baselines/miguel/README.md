# Miguel's mean-field and GAMP baselines

Both models have independent Bernoulli latents, a linear Gaussian decoder,
learned diagonal observation variance, bias, and prior probabilities. Inference
holds the learned prior fixed. Training does not differentiate through inference.

- [`mf.py`](mf.py): parallel damped mean-field inference, an Adam dictionary
  update on the fixed-posterior ELBO, followed by sequential moment updates.
- [`gamp.py`](gamp.py): damped diagonal sum-product GAMP, an Adam dictionary
  update using the approximate factor-belief Fisher score, and simultaneous
  moment targets. Its covariance correction is part of the method.
- [`config.py`](config.py): every default and selected benchmark recipe.
- [`shared.py`](shared.py): data streams, training, checkpointing, and evaluation.

Parameter `w` has shape `[width, input_dim]`, `b` and `logvar` have input width,
and `logits` has dictionary width. `probabilities(parameters, x, config)` is the
common inference interface. Readout thresholds probabilities at 0.5;
reconstruction uses posterior means.

## Environments

Install the main project in each relevant environment with `pip install --no-deps -e .`.
The numerical reference training stack is pinned in
[`requirements-train.txt`](requirements-train.txt): JAX/JAXlib 0.9.2,
Optax 0.2.8 and NumPy 2.4.4. For a Linux CUDA training environment:

```bash
pip install -r baselines/miguel/requirements-train.txt
pip install --no-deps -e .
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=false
export CUDA_VISIBLE_DEVICES=0
```

The method functions also work with CPU JAX. Presets initialize width 16K Gemma
weights on CPU and wider/synthetic weights on GPU, preserving the source's
initialization arithmetic. A custom JSON can select `init_backend="cpu"`.
Training uses one visible JAX device per run. CUDA evaluation uses one visible
GPU. Different hardware and compiler versions need not be bitwise identical;
the runtime records actual versions and initialization hashes without requiring
private reference artifacts.

For SAE Probes and RAVEL, use the optional pinned scoring environment from
[`requirements-score.txt`](requirements-score.txt). The source environment used
an override for the older TransformerLens Beartype constraint; with `uv`:

```bash
uv pip install --override baselines/miguel/requirements-overrides.txt \
  -r baselines/miguel/requirements-score.txt
uv pip install --no-deps -e .
```

Synthetic observations and metrics run in a separate PyTorch process using the
main project's synthetic dependencies. Install this repository in that environment
as well; pass its Python executable through `--sampler-python`. The bridge sends
only observations to training. Ground-truth features are used only for scoring.

## Explicit presets

| Dataset | Method | Width | Dictionary LR | Moment LR | Damping |
| --- | --- | --- | --- | --- | --- |
| Gemma raw, 50M | MF | 16K / 65K | 0.0003 | 0.01 | first 1, then 0.2 |
| Gemma raw, 50M | GAMP | 16K / 65K | 0.001 / 0.003 | 0.001 / 0.003 | 0.5 |
| Synthetic, 200M | MF | 16K | 0.001 | 0.003 | first 1, then 0.2 |
| Synthetic, 200M | GAMP | 16K | 0.03 | 0.03 | 0.5 |

All use 20 inference iterations and cosine LR decay to 10% of the peak.
Gemma batch size is 2048; synthetic batch size is 1024. Token budgets round down
to complete batches. `--training-tokens 500000000` selects the source's 500M Gemma
rates. `--whiten` selects its 50M MF recipe for widths 16K, 65K, 131K and 524K,
including its width-scaled prior, norm cap, clipping and batch skipping.
Preset seeds are 0–4; edit the emitted configuration for other experiments.

```bash
python -m baselines.miguel.mf config --width 16384 --seed 0 --whiten \
  --output configs/mf-gemma.json
python -m baselines.miguel.mf train --config configs/mf-gemma.json \
  --cache data/gemma --normalization data/gemma/normalization \
  --output outputs/mf-gemma
```

Prepare raw Gemma activations with `python -m experiments.gemma.prepare`,
`python -m experiments.gemma.extract`, and `python -m experiments.gemma.normalize`
(see the main experiments documentation for arguments). Whitening reads `mean.npy`, `whitener.npy`,
and `dewhitener.npy` from the same normalization directory used by BeFOND.
Omit `--normalization` for raw-input configurations.

```bash
python -m baselines.miguel.gamp config --target synthsaebench \
  --width 16384 --seed 0 --output configs/gamp-synthetic.json
python -m baselines.miguel.gamp train --config configs/gamp-synthetic.json \
  --snapshot data/synthetic/world --sampler-python /path/to/data-env/bin/python \
  --output outputs/gamp-synthetic
python -m baselines.miguel.gamp synthetic-eval --run outputs/gamp-synthetic \
  --snapshot data/synthetic/world --sampler-python /path/to/data-env/bin/python \
  --rare --output outputs/gamp-synthetic/evaluation.json
```

The public synthetic world is `decoderesearch/synth-sae-bench-16k-v1` at revision
`b2efd8b919ae46d6d487c73d46db5ee52813621d`, using the historical parent-unscaled
configuration. `befond.data.synthetic_spec.download_snapshot()` downloads that
world. Synthetic training uses 199,999,488 samples, data seed 1, and disjoint keyed
training/test streams. MCC uses one-to-one absolute-cosine matching; the baseline
F1 matches each learned atom to its maximum-absolute-cosine ground-truth feature.
`--rare` additionally reports the shared positive-cosine ground-truth metrics.

## Checkpoints and evaluation

A run saves its full configuration, hashes/version metadata, parameter-only
`checkpoint.npz`, and resumable `state-TOKENS.npz` files. The NPZs contain arrays
and JSON metadata, never executable pickles. Use `--stop-tokens` for a partial run
without shortening its cosine horizon, then `--resume` with the saved state and
the same output directory. Partial stops must align with eight-batch buffers.
`--checkpoint-every` also retains parameter snapshots at buffer boundaries.

```bash
python -m baselines.miguel.mf sae-probes --run outputs/mf-gemma \
  --cache data/sae-probes --normalization data/gemma/normalization \
  --output outputs/mf-gemma/probes
python -m baselines.miguel.mf ravel --run outputs/mf-gemma \
  --cache data/ravel --normalization data/gemma/normalization \
  --output outputs/mf-gemma/ravel
```

SAE Probes generates any missing last-token activations for its 113 tasks through
the shared BeFOND preparation code. It then reuses the cache on later runs.
RAVEL also prepares its model-only dataset cache from public sources. The adapter retains natural binary, norm-scaled codes and the
original raw-coordinate decoder after whitening. These protocols use the pinned
SAEBench revision `8042bb3828c6340da8d12062324e92b2077c571c` and Gemma revision
`c5ebcd40d208330abc697524c919956e692655cf`.
