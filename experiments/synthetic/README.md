# Synthetic dictionary recovery

This experiment learns a Bernoulli sparse-coding dictionary from the pinned public SynthSAEBench generator. Start with [`configs/synthetic.json`](../../configs/synthetic.json); it contains the current model, training, data, and evaluation settings.

The generator contains 16,384 features in 768 dimensions. `benchmark: historical` selects the source project's distribution with **parent amplitude scaling disabled**. `published` enables that scaling; it changes the data distribution, so record the choice when comparing results.

Use the synthetic environment described in the root README, then run from the repository root:

```bash
python -m experiments.synthetic.prepare --device cuda:0
python -m experiments.synthetic.train --device cuda:0 --output outputs/synthetic
python -m experiments.synthetic.evaluate CHECKPOINT --device cuda:0 --output outputs/synthetic/metrics.json
```

`CHECKPOINT` is a checkpoint file or exported model directory. Evaluation reads its saved configuration. Supply `--config configs/synthetic.json` when evaluating a model bundle without training metadata.

Preparation downloads and hash-checks the public generator. The current defaults fit whitening on a separate stream of **2 billion examples**, in batches of 50,000. This is a large preparation job. For a small integration check, use a separate configuration/output path and override `--samples` and `--batch-size`; this produces a different normalization estimate. The account-free toy example in the root README does not need this dataset or preparation.

Current training uses 250,000 batches of 2,048 examples, plus separate prior-fitting batches every ten updates. The default inference horizon is exponentially sampled with mean 32 and capped at 100. Default final evaluation uses 20 inference steps and 500,000 held-out examples. These are the current code defaults, not a frozen historical paper recipe.

The sampler derives every batch from `(data seed, stream, step)`, so evaluation and prior fitting do not consume the training random stream. The separate 500,000-example frequency reference defines rare features before scoring.

Read the implementation in this order:

1. [`befond/data/synthetic.py`](../../befond/data/synthetic.py): generator loading and independent batches.
2. [`befond/training.py`](../../befond/training.py): inference followed by learning.
3. [`befond/evaluation/synthetic.py`](../../befond/evaluation/synthetic.py): streamed evaluation.
4. [`befond/evaluation/feature_metrics.py`](../../befond/evaluation/feature_metrics.py): positive-cosine one-to-one matching and frequency-stratified detection.

MCC uses absolute-cosine one-to-one dictionary alignment. Macro F1 averages support detection over every ground-truth feature, with unmatched features scored zero. Rare F1 restricts that average to the least frequent quarter. Support uses posterior probability greater than 0.5. Reconstruction MSE uses the full posterior mean in the model's input coordinates. `--samples`, `--frequency-samples`, `--steps`, and `--batch-size` allow smaller evaluation checks; record those changes with the results.
