# Comparison models

The baselines live alongside BeFOND so experiments do not require a private
research checkout. They keep their own objectives and optimizers.

| Package | Models | Start here |
| --- | --- | --- |
| [`sae`](sae/README.md) | ReLU, Gated, TopK, BatchTopK, JumpReLU, Matryoshka | Small PyTorch models and a shared-data training example |
| [`miguel`](miguel/README.md) | Bernoulli mean field and GAMP | Independent inference and learning rules; Gemma and synthetic recipes |
| [`david`](david/README.md) | Original and improved full-width BatchTopK, Matryoshka, JumpReLU | Complete synthetic training recipes, evaluation, and published checkpoint loading |

`sae` needs only the main BeFOND dependencies. Miguel's implementations use JAX;
David's implementations use a specific SAELens revision. Keep those optional
runtime environments separate, following each package's requirements file.
Importing `baselines` does not import either stack.

The current model and optimizer defaults are explicit in
[`sae/config.py`](sae/config.py), [`sae/training.py`](sae/training.py),
[`miguel/config.py`](miguel/config.py), and [`david/recipes.py`](david/recipes.py).
Each training command saves its resolved configuration next to the weights.

The compact `sae` models are the implementations from this project's SAE code.
The David package preserves the supplied full-width synthetic study code.
David's exact Gemma training code is pending inclusion; the compact SAE
implementations should not be described as reproducing those separate Gemma runs. Gemma data preparation and evaluation are available under
`befond.data` and `befond.evaluation` for use with released compatible models.

## Verification

Run each optional stack in its own environment:

```bash
pytest baselines/sae/tests -q
pytest baselines/miguel/tests -q
pytest baselines/david/tests -q
```

The compact SAE checks cover all six architectures, portable checkpoint round
trips, and exact continuation after an interrupted keyed training run. Miguel's
check covers the analytic single-latent mean-field posterior and finite updates
for both methods. David's checks cover gates, training continuation, resampling,
inference export, and all catalog experiment plans.
