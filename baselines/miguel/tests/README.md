# Mean-field checks

[`test_methods.py`](test_methods.py) compares the single-latent mean-field
posterior with its analytic solution, then checks one parameter update for
mean field. Parameters must stay finite and variances positive.

Use an environment with the project, pytest, JAX and Optax
(`python -m pip install -e ".[test,miguel]"`). Run **from the repository root**:

```bash
JAX_PLATFORMS=cpu python -m pytest baselines/miguel/tests -q
```

The test uses small generated arrays and compiles the updates on CPU. It needs
no dataset or model downloads. Pytest skips this file if JAX or Optax is missing;
install the extra above to run the numerical check. See the [package guide](../README.md)
for pinned experiment environments and benchmark commands.
