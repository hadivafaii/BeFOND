# Model bundles and future checkpoint releases

Each BeFOND width is independent. A Hub repository may contain subfolders such
as `gemma/16384/` and `gemma/65536/`; these are layout examples, not published
release identifiers. Each folder contains:

```text
config.json          # Input dimension, width, inference settings, fitted parameters
model.pt             # Tensor-only model state; loaded with weights_only=True
metadata.json        # Schema, model identity, training/evaluation provenance
normalization.pt     # Optional frozen transform from raw to training coordinates
```

`save_pretrained(model, directory, metadata=..., normalization_path=...)` creates
a bundle. `from_pretrained(path_or_repo, subfolder=..., revision=...)` loads one.
Only the requested width's files are downloaded. Use a fixed Hub commit for
reproducible evaluation. No token, W&B access, source checkout, or remote Python
code is needed to load a public bundle.

The model's `infer` method takes observations in training coordinates. A loaded
model exposes `normalization_path` and `pretrained_metadata`. The benchmark
adapter applies that saved transform to raw activations and maps decoder atoms
and reconstructions back to raw coordinates. Direct users can load it with:

```python
import torch
from befond.data.normalization import ActivationNormalizer

normalizer = ActivationNormalizer.from_state_dict(
    torch.load(model.normalization_path, map_location="cpu", weights_only=True))
posterior = model.infer(normalizer.transform(raw_x), steps=100)
raw_reconstruction = normalizer.inverse(model.reconstruct(posterior.mean))
```

For models trained without normalization use `ActivationNormalizer.none()`.

## Convert existing research fits

Run conversion where the trusted fit files are available. It does not import
the research package and accepts only the Bernoulli NGD case:

```bash
befond-convert /path/to/ckpt_latest.pt \
  --model-config /path/to/ConfigPoisVAE.json \
  --training-config /path/to/ConfigTrain.json \
  --normalization /path/to/normalization.pt \
  --output outputs/converted-width
```

Older private fits may contain Python objects requiring `--trusted`. Only use
that option for your own trusted files. Converted/public model weights use the
restricted tensor loader. Conversion transposes decoder atoms into `[D,K]` and
converts stored log standard deviation into log variance. Width and input
dimension are inferred from the actual weight tensor.

`--normalization` also accepts the original normalization directory. Select its
mode with `--normalization-mode whiten` (the default) or the matching scalar
normalization mode. Conversion saves a portable tensor transform inside the
bundle. Clamped, stochastic, and non-Bernoulli research configurations are
rejected rather than silently changing inference.

Verify a converted model against its original inference outputs and include its
normalization before publication. Publishing the files is a separate action;
the loader does not upload or create Hub repositories.

## Training continuation

`checkpoint.pt` stores the completed step count, model and training settings,
random state, and any data-source state. Continuation checks model, data, and
training-critical settings. Width cannot change during resume. A pretrained
inference bundle is not a replacement for a training checkpoint.

Miguel, David, and the compact SAE family retain their own model formats and
loaders under `baselines/`, because their parameters and optimizers differ.
