# Guides

These guides explain the model and how to move trained weights between training
and inference. Start with the algorithm if you want to follow the code; start
with checkpoints if you want to use or release a trained model.

- [`algorithm.md`](algorithm.md): Bernoulli inference, natural-gradient learning,
  update order, tensor shapes, and the main hyperparameters.
- [`checkpoints.md`](checkpoints.md): portable model bundles, normalization,
  conversion of existing research fits, and training continuation.

After installing from the repository root, inspect the training settings and
checkpoint converter without starting a run:

```bash
pip install -e '.[test]'
python examples/train_toy.py --show-config
befond-convert --help
```

For a small CPU check of the equations described in the algorithm guide:

```bash
python -m pytest -q tests/test_math.py
```

See [`befond/`](../befond/README.md) for the implementation map and
[`examples/`](../examples/README.md) for a complete small training run.
