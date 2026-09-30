# Gemma activations and SAEBench

This path learns a BeFOND dictionary on the layer-12 post-residual activations of Gemma-2-2B. [`configs/gemma.json`](../../configs/gemma.json) holds current defaults. The model, tokenizer, corpus, and benchmark revisions are pinned in [`befond/data/gemma_spec.py`](../../befond/data/gemma_spec.py).

Use the separate Gemma environment from the root README. Obtain access to `google/gemma-2-2b` and authenticate Hugging Face in your own environment, then run:

```bash
python -m experiments.gemma.prepare --data-dir data/gemma
python -m experiments.gemma.extract --device cuda:0 --data-dir data/gemma
python -m experiments.gemma.normalize --device cuda:0 --data-dir data/gemma
python -m experiments.gemma.train --device cuda:0 --output outputs/gemma
python -m experiments.gemma.evaluate CHECKPOINT --device cuda:0 --output outputs/gemma/evaluation
```

`CHECKPOINT` is a checkpoint file or exported model directory. Saved training metadata supplies evaluation and normalization settings. Use `--config configs/gemma.json` for a converted bundle that lacks these settings.

Preparation reserves disjoint document ranges for 65,536 validation tokens, 65,536 calibration tokens, and 500 million training tokens from Pile Uncopyrighted. It tokenizes each document separately with BOS and at most 1,024 tokens, without packing; BOS and padding activations are excluded. Extraction stores native BF16 activations in verified, resumable shards.

The full training activation cache occupies about **2.30 TB**, before model weights, token files, evaluation caches, and checkpoints. Whitening uses all prepared training tokens. Normalization accumulation can resume from its saved progress. Current training uses the first 50 million tokens, a global batch of 4,096, 12,207 updates, 16,384 dictionary atoms, and a maximum training inference horizon of 100. The default evaluation uses 100 inference steps. These are the current code defaults.

For a bounded preparation check, `prepare` accepts `--training-tokens`, `--validation-tokens`, and `--calibration-tokens`. Use a separate data directory and update the configuration's data paths and token budget. Reduced preparation changes the fitted normalization and is not the full default experiment.

The evaluation command runs SAE Probes and RAVEL. To run one family or a small probe subset:

```bash
python -m experiments.gemma.evaluate CHECKPOINT --device cuda:0 \
  --output outputs/gemma/probes --evals sae_probes
python -m experiments.gemma.evaluate CHECKPOINT --device cuda:0 \
  --output outputs/gemma/short --evals sae_probes --sae-probes-datasets DATASET_NAME
```

Upstream SAE Probes defines valid dataset names and feature selection, probe fitting, and scoring. Raw language-model activations are cached under `data/gemma/evaluation`. RAVEL uses the pinned Hugging Face model and dataset, keeps gradients through the intervention mask and linear decoder, and aggregates only after every configured entity completes. Reusing an output directory resumes its completed evaluations; changing inference settings requires another output directory.

[`befond/evaluation/adapter.py`](../../befond/evaluation/adapter.py) is the central interface: it whitens raw activations, performs Bernoulli inference, applies `p * (p > 0.5)`, and rescales codes by raw decoder norms. Its decoder maps back into the language model's original activation coordinates. The adapter freezes BeFOND parameters but preserves decoder gradients needed by RAVEL.
