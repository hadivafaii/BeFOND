# Superposition

This experiment varies input dimension while keeping 16,384 generating features and a learned dictionary of 4,096 atoms. The current search compares one inference step with an exponentially sampled horizon of mean 5.5, capped at 50.

Prepare dimension-specific generators, then run the explicit 768-dimensional starting configuration:

```bash
python -m experiments.superposition.prepare --output data/superposition --device cuda:0
python -m experiments.superposition.train --device cuda:0 --output outputs/superposition
python -m experiments.superposition.evaluate CHECKPOINT --device cuda:0 --output outputs/superposition/metrics.json
```

Preparation uses the checked-in generator recipe and SAE-Lens 6.49.1, verifies generated snapshots, and reuses completed dictionaries. Inputs are unwhitened. The base configuration initializes decoder columns to norm 2, then clips them at norm 4. It uses a 100,000-update schedule and stops after 8,000 updates, preserving the current pilot schedule. To continue the full schedule, change `train.stop_after_updates` while retaining the original learning-rate horizon.

Generate the current grid as individual, readable configurations:

```bash
python -m experiments.superposition.sweep --output outputs/superposition/configs
python -m experiments.superposition.train \
  --config outputs/superposition/configs/d768_lr0.001_beta0.5_exp-5.5.json \
  --device cuda:0 --output outputs/superposition/d768
```

[`configs/superposition_sweep.json`](../../configs/superposition_sweep.json) records dimensions 256–1,536, learning rates, beta values, inference depths, and dimension-specific initialization norms. It is a grid description, not a directly executable training configuration. The single base configuration chooses the first learning rate and beta and the recurrent horizon; it is not a claim that these are optimal settings.

Evaluation shares the synthetic metric implementation. Current pilot defaults use 32,768 held-out examples and 100 inference steps. Use a separate, sufficiently large test set for final comparison; do not select settings using its scores.
