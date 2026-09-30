# Attribution and dependency notices

The original BeFOND implementation and this repository's changes are MIT
licensed. Dependencies and upstream model/data artifacts retain their licenses.

- Miguel's mean-field and GAMP implementations are preserved in
  `baselines/miguel/`, including their numerical update rules.
- David's supplied synthetic SAE study is preserved in `baselines/david/`.
  Its source revisions and original file hashes are recorded in
  `baselines/david/provenance.json`. The Decode Research coefficient controller's
  original MIT notice is in `baselines/david/LICENSE.autotuner`.
- SynthSAEBench uses the public Decode Research generator and benchmark
  implementation. Generator identities, hashes, and revisions are recorded in
  `befond/data/synthetic_spec.py`.
- Gemma, the token datasets, SAE-Lens, SAEBench, and SAE Probes are downloaded or
  installed separately. Their use follows the respective upstream terms and
  access requirements. Model weights and benchmark datasets are not vendored.

The compact SAE implementations in `baselines/sae/` come from the development
repository and expose their own model objectives. They are distinct from the
supplied David study recipes.
