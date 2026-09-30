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
         -Wr-\tfrac12\operatorname{diag}(W)\odot(1-2r)
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

ETD1 advances `u` by `gain * u_dot`, with
`gain = (1-exp(-beta*h))/beta`. This handles the linear relaxation exactly while
freezing the nonlinear drive for one interval. The `h=inf` limit has gain
`1/beta`; it does not imply that recurrent inference has converged.

Start with [`inference.py`](../befond/inference.py), then the short
`inference_step` and `infer` methods in [`model.py`](../befond/model.py).

## Learning

At the last inference step, learning detaches the posterior and forms the
expected second moment

```math
E=\langle\operatorname{diag}[r\odot(1-r)]+rr^T\rangle.
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
| `model.inference_h` | Integration interval; combines with `kl_beta` to determine ETD1 gain |
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
