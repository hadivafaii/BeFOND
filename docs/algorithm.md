# Reading the algorithm

An observation is a vector `x` of length `D`. The model has `K` binary latent
variables and a dictionary `decoder` with shape `[D, K]`. A decoder column is
one feature. The likelihood is Gaussian with mean `bias + decoder @ z` and
diagonal observation variance.

## Inference

Inference maintains one logit `u` per latent and per input. Its Bernoulli mean
is `r = sigmoid(u)` and its variance is `r * (1-r)`. The implementation computes
these moments stably at extreme logits. `r` is a probability, not a sampled or
thresholded binary code.

Let `Phi` denote the decoder and `W = Phi.T @ precision @ Phi`. The natural
gradient of free energy gives

```math
\dot u = \Phi^T\Sigma^{-1}(x-b)
         -Wr-\tfrac12\mathrm{diag}(W)\odot(1-2r)
         -\beta(u-u_{\mathrm{ref}}).
```

The code applies the dictionary twice to evaluate the recurrent term without
materializing `W` for a wide dictionary. The Bernoulli uncertainty term matters:
using the mean code alone would define a different learning problem.

`inference_prior_retention` controls how the incoming posterior is mixed with
the baseline prior. Retention 1 uses the incoming posterior as the next
reference; retention 0 uses the baseline prior. Mixing may take place in logit
or probability coordinates. Each outer step fixes its reference while its inner
steps refine the posterior. Every new input starts independently at the learned
baseline prior.

ETD1 proposes advancing `u` by `gain * u_dot`, with
`gain = (1-exp(-beta*h))/beta`. This handles the linear relaxation exactly while
freezing the nonlinear drive for one interval. The `h=inf` limit has gain
`1/beta`; it requires `beta > 0` and does not imply that recurrent inference has
converged. For `beta=0` and finite `h`, the gain is `h`.

`model.inference_integrator` selects the update:

- **`etd1_guarded` (default for all datasets):** start with the ETD1 gain and
  backtrack until the candidate passes a free-energy decrease check for every
  example in the batch. One scalar gain is shared across the batch.
- **`etd1`:** accept the ETD1 proposal without a free-energy check, as in the
  recorded historical runs.
- **`euler`:** advance with gain `h`, which must be finite.

The guard evaluates the expected reconstruction energy, including Bernoulli
uncertainty, plus `beta * KL(q || reference)`. The reference and model
parameters stay fixed throughout the search; terms constant in `u` are omitted.
For each example, with direction `d = u_dot`, the acceptance condition is

```math
E(u+g d) \le E(u)
  -10^{-4}g\sum_k r_k(1-r_k)d_k^2
  +10^{-6}(1+|E(u)|).
```

The candidate energy must also be finite. The search tries the initial gain, then up
to 20 halvings, and raises `RuntimeError` if no candidate passes. These energy
evaluations add computation; `inference_h` sets the initial proposal, and the
accepted gain may be smaller. The check applies to the current fixed reference,
so it does not guarantee decreasing energy across outer steps whose references
change. Guarded inference requires `model.t_inner=1` and rejects an
`inner_steps` override other than 1. `InferenceResult.inference_gain` records the
accepted gain from the last step.

Because every example shares the accepted gain, batch composition can change
an example's inference trajectory. Keep batch and microbatch sizes fixed when
comparing results. Distributed training synchronizes acceptance across workers
so that one gain applies to the global batch. Evaluation checks each inference
batch independently; Gemma's adapter applies the guard separately to each
microbatch and records its effective size with the evaluation outputs.

Start with [`inference.py`](../befond/inference.py), then the short
`inference_step` and `infer` methods in [`model.py`](../befond/model.py).

## Learning

At the last inference step, learning detaches the posterior and forms the
expected second moment

```math
E=\langle\mathrm{diag}[r\odot(1-r)]+rr^T\rangle.
```

The decoder's frozen-statistics natural-gradient flow uses this posterior
moment and the second moment of its reference distribution. The numerical
solver applies the associated matrix exponential without constructing a dense
`K x K` matrix when a matrix-free solver is selected. Posterior variance is
retained in these moments; replacing it with only `r.T @ r` changes the model.

The decoder and enabled observation parameters update **once per training
minibatch**, after the sampled inference horizon. The code then applies the
configured decoder norm constraint. Learned bias uses the source
implementation's one-inner-step inference-response correction.

The baseline prior has a separate update cadence and inference stream. Its
longer rollout can stop when posterior movement remains small. Unchecked or
unstable targets are rejected. Accepted target probabilities are exponentially
averaged into the baseline prior, then projected onto the requested floor and
mean-probability cap.

[`learning.py`](../befond/learning.py) shows the parameter updates.
[`training.py`](../befond/training.py) shows their order. The solver internals in
[`solvers.py`](../befond/solvers.py) are independent of dataset preparation.

## Hyperparameters to inspect first

| Field | Meaning |
| --- | --- |
| `model.num_latents` | Dictionary width; independent of observed dimension |
| `model.t_outer`, `train.horizon_dist` | Maximum inference depth and distribution sampled during training |
| `model.inference_integrator` | `etd1_guarded` (default), `etd1`, or `euler` |
| `model.t_inner` | Inner steps per fixed reference; must be 1 for guarded ETD1 |
| `model.inference_h` | Initial integration interval; combines with `kl_beta` to determine the ETD1 proposal gain |
| `model.inference_prior_retention` | How much of the incoming posterior remains in the next reference |
| `train.kl_beta` | Weight on KL to the inference reference |
| `train.lr`, `lr_min`, `scheduler_type` | Decoder learning-rate schedule; other rates use explicit multipliers |
| `train.r0_budget` | Cap on mean baseline activation probability, not total expected active count |
| `train.r0_floor` | Minimum baseline probability for each latent |
| `train.decoder_norm_target` | Target atom norm under the chosen constraint |
| `train.ngd_prior_*` | Prior update cadence, batch budget, and stopping criteria |
| `train.ngd_decoder_solver` | Dense or matrix-free evaluation of the same configured decoder flow |
| `evaluation.validation_steps`, `steps` | Periodic validation depth and final benchmark depth |

For a baseline prior, `num_latents * mean(prior)` is its expected active count.
Posterior activity on real data can differ. Changing width does not automatically
rescale the other hyperparameters.
