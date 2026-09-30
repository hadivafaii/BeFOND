# Experiments

This package connects the shared BeFOND model, trainer, and metrics to three datasets. Each experiment has a small training entrypoint and its own preparation/evaluation commands; the numerical learning rule stays in [`befond/`](../befond/README.md).

With the base package installed, run **from the repository root**:

```bash
python -m experiments.synthetic.train --show-config
python -m experiments.gemma.train --show-config
python -m experiments.superposition.train --show-config
```

These commands print resolved settings without downloading data or running training. Add `--help` to an entrypoint to inspect its options.

| Directory | What it contains |
| --- | --- |
| [synthetic](synthetic/README.md) | Dictionary and feature recovery on the public SynthSAEBench generator. |
| [superposition](superposition/README.md) | Dimension-specific generators and a grid comparing inference horizons as feature overlap changes. |
| [gemma](gemma/README.md) | Gemma activation preparation, whitening, training, SAE Probes, and RAVEL. |

Start with the JSON files in [`configs/`](../configs/README.md), then read the experiment's `train.py`, `prepare.py`, and `evaluate.py`. Training entrypoints delegate to [`befond/training.py`](../befond/training.py); data sources and metrics live in [`befond/data`](../befond/data/README.md) and [`befond/evaluation`](../befond/evaluation/README.md).

Full synthetic/superposition runs need the optional `synthetic` environment. Gemma uses the separate optional `gemma` environment and Hugging Face model access. Follow each experiment's README before launching its preparation or training recipe; the default budgets are substantial. For a small training example without downloaded data, use [`examples/`](../examples/README.md).
