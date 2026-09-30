# Small examples

Start here to see a complete BeFOND training run. [`train_toy.py`](train_toy.py)
uses [`toy.json`](toy.json) to learn 24 binary features from 12-dimensional
synthetic observations. It runs 30 updates on CPU, without downloads or accounts.

Run these commands from the repository root:

```bash
pip install -e '.[test]'
python examples/train_toy.py --show-config
python examples/train_toy.py --output outputs/toy
```

The `--show-config` command prints the resolved hyperparameters. The training command
writes metrics to `outputs/toy/metrics.jsonl`, a resumable `checkpoint.pt`, and
an inference bundle in `pretrained/`. Use a new output directory when starting
another run.

To check that training, saving, and resuming work:

```bash
python -m pytest -q tests/test_training.py
```

These small settings are for learning the interface. See
[`configs/`](../configs/README.md) for experiment defaults and
[`experiments/`](../experiments/README.md) for benchmark instructions.
