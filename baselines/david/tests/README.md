# Full-width study checks

[`test_sweep.py`](test_sweep.py) tests the custom MinFire gate, resumable training,
dead-latent resampling, and export to an ordinary inference SAE. It also checks
that every bundled catalog entry produces the intended experiment command.

Use the dedicated SAELens 6.51 environment described in the
[package installation guide](../README.md#install), and install `pytest` there.
Run **from the repository root**:

```bash
python -m pytest baselines/david/tests -q
```

These are small CPU checks using generated tensors and temporary checkpoints.
They do not download the synthetic world or published weights. To check only
the bundled experiment plans in that same environment:

```bash
python -m pytest baselines/david/tests/test_sweep.py -k catalog -q
```
