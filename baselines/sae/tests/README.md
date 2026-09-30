# Compact SAE checks

[`test_models.py`](test_models.py) checks all six architectures, one optimizer
update, decoder normalization, and exact checkpoint round trips. It also checks
that an interrupted run resumes with the same parameters as uninterrupted
training, including the optimizer and learning-rate schedule.

Install the project and test extra (`python -m pip install -e ".[test]"`), then
run **from the repository root**:

```bash
python -m pytest baselines/sae/tests -q
```

All tests use CPU tensors and temporary output directories. No dataset, Hub
checkpoint, GPU, or W&B account is needed. Architecture implementations are in
[`../models`](../models/README.md).
