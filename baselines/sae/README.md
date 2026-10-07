# Small PyTorch SAEs

This package implements six feedforward sparse autoencoders, with a common
training loop and checkpoint format. It is a small starting point for comparing
encoder architectures on the same data as BeFOND.

With the project and test dependencies installed
(`python -m pip install -e ".[test]"`), run **from the repository root**:

```bash
python -m pytest baselines/sae/tests -q
```

The checks run on CPU without downloading data or pretrained models. They cover
all architectures, checkpoint loading, and continuation after interrupted training.

Read [`config.py`](config.py), then the selected file in [`models`](models),
then [`training.py`](training.py). The architecture code is preserved from the
research implementation. The smaller trainer removes private datasets, sweep
orchestration, plotting, and W&B requirements while keeping the loss, Adam
update, decoder normalization, dead-feature bookkeeping and learning-rate rule.

| Model type | Objective |
| --- | --- |
| `relu` | Per-example reconstruction SSE + activation L1 |
| `gated` | Gated loss, including detached auxiliary reconstruction |
| `topk` | TopK fraction of variance unexplained + dead-latent AuxK |
| `batch_topk` | BatchTopK FVU + AuxK; running threshold at inference |
| `jumprelu` | Reconstruction SSE + learned-threshold L0 penalty |
| `matryoshka` | Mean prefix FVU + AuxK |

`num_latents` is the full dictionary width; `k_active` is the activity budget.
Decoder rows have shape `[num_latents, input_dim]`. Encoder alignment is also
available for ReLU, TopK, and BatchTopK with `aligned=True`.

Run the small example:

```bash
python -m baselines.sae.train --config baselines/sae/configs/toy.json \
  --output outputs/sae-toy --device cpu
```

Change `model_type` in that JSON to run another architecture. All default model
hyperparameters are listed in [`SAEConfig`](config.py), and optimizer defaults
in [`TrainingConfig`](training.py). The demonstration deliberately uses a shorter
training horizon; the source optimizer defaults are 12,000 steps, batch 200,
Adam at 1e-3, 1,000 warmup steps, and cosine decay to 1e-5.

The `data` object follows the main BeFOND configurations, so synthetic worlds
and cached Gemma activations use the same sampling and normalization code.
For those datasets, set `model.input_shape` to `[768]` or `[2304]` and select the
appropriate dictionary width. Keep training hyperparameters explicit for each
comparison; the toy configuration is not a Gemma reproduction recipe.

Training saves `run.json`, `history.jsonl`, `config.json`, `model.safetensors`,
and an atomic `training.pt` with Adam, schedule and dead-feature state. Resume
with the same command plus `--resume`. Sampling resumes from the next keyed batch.
`training.pt` is for local resumption; the safetensors/config pair is portable.

```python
from baselines.sae.checkpoints import load_sae

sae = load_sae('outputs/sae-toy')
# Or, once published: load_sae('owner/repository', subfolder='topk/width-16384')
codes = sae.encode(x)
reconstruction = sae.decode(codes, flatten=True)
```

The model consumes data in its training coordinates. If the data configuration
uses whitening or scaling, apply that same `ActivationNormalizer` before encoding
and its `inverse()` after decoding; retain the normalization directory with the
checkpoint release.
