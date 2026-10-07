# Superposition

This experiment varies input dimension while keeping 16,384 generating features and a learned dictionary of 4,096 atoms. The base configuration uses an exponentially sampled inference horizon of mean 5.5, capped at 50.

Inference defaults to `etd1_guarded`, including the `inference_h: "inf"` setting in the base configuration. See the [algorithm guide](../../docs/algorithm.md#inference) for the acceptance rule and optional integrators.

For a quick check, use the base installation and run these commands **from the repository root**:

```bash
python -m experiments.superposition.train --show-config
```

This prints the configuration without generating data or starting training. [`prepare.py`](prepare.py) creates the dictionaries, and [`train.py`](train.py) uses the shared trainer. [`evaluate.py`](evaluate.py) reuses the synthetic evaluation command.

Use the optional `synthetic` environment from the [root README](../../README.md) for the full experiment. Prepare dimension-specific generators, then run the explicit 768-dimensional starting configuration **from the repository root**:

```bash
python -m experiments.superposition.prepare --output data/superposition --device cuda:0
python -m experiments.superposition.train --device cuda:0 --output outputs/superposition
python -m experiments.superposition.evaluate CHECKPOINT --device cuda:0 --output outputs/superposition/metrics.json
```

Preparation uses the checked-in generator recipe and SAE-Lens 6.49.1, verifies generated snapshots, and reuses completed dictionaries. Inputs are unwhitened. The base configuration initializes decoder columns to norm 2, then clips them at norm 4. It uses a 100,000-update schedule and stops after 8,000 updates, preserving the current pilot schedule. To continue the full schedule, change `train.stop_after_updates` while retaining the original learning-rate horizon.

The base file is a development preset, not a selected paper configuration.

Evaluation shares the synthetic metric implementation. Current pilot defaults use 32,768 held-out examples and 100 inference steps. Use a separate, sufficiently large test set for final comparison; do not select settings using its scores.
