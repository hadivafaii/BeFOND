# Selected Gemma paper models

[`paper_models.json`](paper_models.json) maps the selected Gemma comparison to
model configurations and source run/checkpoint identities. It contains the
models selected for sparse probing and RAVEL, without the pilot search results.
When probing and RAVEL select different sparsities, both models are identified.

## BeFOND

The twelve configurations below preserve the recorded model and training
settings for seeds 0–2 except that runnable recipes now select `etd1_guarded`
as the inference integrator. The original runs used `etd1`; pass
`--set model.inference_integrator='"etd1"'` when training to select that original
update rule. Each recipe uses 12,207 decoder updates with global batch 4,096
(49,999,872 tokens), and whitening fitted on 500M tokens.

| Width | Seed 0 | Seed 1 | Seed 2 |
| ---: | --- | --- | --- |
| 16,384 | [config](../../configs/gemma-16k-seed0.json) | [config](../../configs/gemma-16k-seed1.json) | [config](../../configs/gemma-16k-seed2.json) |
| 65,536 | [config](../../configs/gemma-64k-seed0.json) | [config](../../configs/gemma-64k-seed1.json) | [config](../../configs/gemma-64k-seed2.json) |
| 131,072 | [config](../../configs/gemma-128k-seed0.json) | [config](../../configs/gemma-128k-seed1.json) | [config](../../configs/gemma-128k-seed2.json) |
| 524,288 | [config](../../configs/gemma-512k-seed0.json) | [config](../../configs/gemma-512k-seed1.json) | [config](../../configs/gemma-512k-seed2.json) |

Final evaluation uses 10 inference steps. Training validation uses 100. The
524K seeds have different recorded initialization, reference-retention and norm
penalty settings, so use the individual files rather than overriding only the
width. Evaluation overrides are recorded separately: the 524K seed-0 model was
trained with reference retention 0.999 and evaluated with retention 1.

```bash
python -m experiments.gemma.train --config configs/gemma-64k-seed1.json --show-config
python -m experiments.gemma.train --config configs/gemma-64k-seed1.json \
  --device cuda:0 --output outputs/gemma-64k-seed1
python -m experiments.gemma.evaluate CHECKPOINT \
  --config configs/gemma-64k-seed1.json --device cuda:0 \
  --output outputs/gemma-64k-seed1/evaluation
```

Configuration provenance includes the source training/evaluation run IDs,
whitening hashes, and recorded runtime versions. Paths use this release's data
layout. These configurations do not promise bitwise reproduction of the
original research runtime or replace the original checkpoint weights.

## BatchTopK and Matryoshka BatchTopK

The index records the exported SAE parameters and training settings for each
selected configuration, plus the five source checkpoints (seeds 1–5). Each
source `training_config.json` is identified by an immutable Hugging Face revision
and SHA-256 hash. The original Gemma MultiSAETrainer code and its full optimizer
configuration are still needed for exact retraining; the exports do not record
optimizer betas or epsilon. The compact models under `baselines/sae` are not
substitutes for that trainer.

| Width | BatchTopK L0: probing / RAVEL | Matryoshka L0: probing / RAVEL |
| ---: | ---: | ---: |
| 16,384 | 400 / 400 | 200 / 400 |
| 65,536 | 400 / 800 | 400 / 400 |
| 131,072 | 600 / 600 | 200 / 400 |
| 524,288 | 800 / 800 | 200 / 400 |

Both use 49,999,872 training tokens, batch 4,096, constant learning rate 0.0003,
FP32 SAEs and SAELens 6.51.1. Model and data revisions, whitening provenance,
initialization and auxiliary-loss settings are in the index. Released inference
weights incorporate whitening and accept raw activations; do not whiten them
again. See the [checkpoint release](https://huggingface.co/chanind-goodfire/baseline-saes)
for loading instructions.

## Mean-field + Adam

The selected whitened recipe is implemented in
[`selected_config`](../../baselines/miguel/config.py), for all four widths and
seeds 0–4. The index records its complete configuration at each width. It uses
20 inference steps, 49,999,872 tokens, dictionary LR 0.0005 and norm cap 1.58.

```bash
python -m baselines.miguel.mf config --width 65536 --seed 1 --whiten \
  --output mf-gemma-64k-seed1.json
```

Follow the [mean-field instructions](../../baselines/miguel/README.md) for its
separate environment and training command.

## Gemma Scope

The index identifies the released checkpoints selected at the four shared
widths in the paper's Gemma Scope table. These keep their original preprocessing
and training budgets. They are evaluated checkpoints, not newly trained models.
