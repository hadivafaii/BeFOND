# Release validation

Validation was performed on September 30, 2026. The source was the current
development checkout at commit `502f587a262f0fb7e2aecbc055a16fb9e5729a8c`.
The earlier anonymous export and research cache were not used to build this
release. The development checkout was left unchanged.

## Numerical and integration checks

The final combined suite passed **45 tests**, including the five optional
research-reference cases. Separate dependency environments passed **one Miguel
test and five David tests**. These are small correctness/integration checks,
not full experimental replications.

- Bernoulli natural gradients agree with gradients of independently enumerated
  free energy. Dense and matrix-free solvers agree with independent matrix
  exponential references across the retained decoder metrics.
- Optional reference tests compare inference trajectories and decoder, bias,
  noise, and prior updates against the research implementation.
- Two complete eight-update training trajectories agree with the research
  trainer, including sampled horizons, learning-rate warmup, rolling reference,
  adaptive prior stopping/rejection, prior constraints, noise freezing, and
  soft decoder norm regularization.
- A two-worker CPU/Gloo check agrees with single-worker updates for uneven
  minibatch and output partitions. A short two-worker training command completes.
- Interrupted training resumes exactly in the deterministic toy example.
  Portable model bundles and legacy conversion preserve inference and precision.
- Normalization preserves raw-coordinate decoder behavior and intervention
  gradients. Existing synthetic and Gemma normalization artifacts load with
  matching transformations. Real pinned synthetic-generator samples and cached
  Gemma minibatches match the research data paths.
- The six compact SAE families pass update and safetensors round-trip tests;
  the compact trainer passes exact continuation checks.
- Miguel's analytic mean-field/finite-update test passes with JAX 0.9.2 and
  Optax 0.2.8. David's five supplied tests pass with the pinned SAE-Lens 6.51
  source revision.

## Packaging and commands

The package was installed in a separate environment. The CPU toy example,
experiment configuration commands, baseline examples, and wheel packaging were
checked. The wheel contains experiment presets, generator metadata, baseline
checkpoint catalogs, and the preserved third-party notice. Benchmark numerical
checks use their separate dependency environments.

Run the ordinary suite with:

```bash
pytest -q tests baselines/sae/tests
```

The research-reference comparisons are optional and skip unless an original
checkout is supplied:

```bash
FONDV2_REFERENCE=/path/to/research pytest -q tests/test_reference_parity.py tests/test_training_reference.py
```

Full-scale model training and complete Gemma/SAEBench evaluation were not rerun
as part of this extraction. GPU throughput and the widest configurations have
not been benchmarked in this new package. BeFOND Hub downloads cannot be tested
against the future releases until those artifacts are uploaded. Local bundle
loading and the selective Hub-loading interface are implemented; the package
does not claim that unreleased weights are available.

David's Gemma-specific training source was not present and remains pending.
