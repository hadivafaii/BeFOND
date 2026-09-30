"""Data-coordinate and stream contracts used by both training and benchmarks."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from befond.model import BeFOND, ModelConfig
from befond.data.normalization import ActivationNormalizer, StreamingCovariance


def test_normalization_bundle_and_raw_decoder(tmp_path):
    from befond.evaluation.adapter import BeFONDAdapter

    torch.manual_seed(4)
    x = torch.randn(127, 4) @ torch.diag(torch.tensor([1., 2., 3., 4.])) + 2
    moments = StreamingCovariance(4, batch_reduction_dtype=torch.float64)
    moments.update(x[:33])
    moments.update(x[33:])
    torch.testing.assert_close(moments.mean, x.double().mean(0))
    centered = x.double() - x.double().mean(0)
    torch.testing.assert_close(moments.m2 / len(x), centered.T @ centered / len(x))
    moments.save(tmp_path / "normalization")
    model = BeFOND(ModelConfig(input_dim=4, num_latents=7, fit_dec_bias=True, t_outer=3))
    for mode in ("whiten", "global_rms", "global_standardize", "center_global_rms", "none"):
        transform = ActivationNormalizer.from_artifact(mode, tmp_path / "normalization")
        torch.testing.assert_close(transform.inverse(transform.transform(x)), x, atol=3e-6, rtol=3e-6)
        torch.save(transform.state_dict(), tmp_path / "normalization.pt")
        restored = ActivationNormalizer.from_artifact(mode, tmp_path / "normalization.pt")
        torch.testing.assert_close(restored.transform(x), transform.transform(x))
        adapter = BeFONDAdapter(model, transform, steps=3, readout="posterior_mean",
                                input_scale=2.0, cfg=SimpleNamespace(d_in=4, d_sae=7))
        codes = adapter.encode(x[:5])
        probability = model.infer(transform.transform(x[:5]) * 2.0, steps=3).mean
        expected = transform.inverse((probability @ model.decoder.T + model.bias) / 2.0)
        torch.testing.assert_close(adapter.decode(codes), expected, atol=3e-6, rtol=3e-6)
        mask_codes = codes.clone().requires_grad_(True)
        adapter.decode(mask_codes).sum().backward()
        assert mask_codes.grad is not None and torch.isfinite(mask_codes.grad).all()


def test_gemma_bf16_shards_and_deterministic_shuffle(tmp_path):
    from befond.data import gemma_spec as spec
    from befond.data.activation_cache import cache_identity, CachedGemmaActivationSource
    from befond.data.gemma import CachedActivationStream

    manifest = dict(schema_version=1, dataset_name=spec.DATASET_NAME,
                    dataset_revision=spec.DATASET_REVISION, tokenizer_name=spec.HF_MODEL_NAME,
                    tokenizer_revision=spec.MODEL_REVISION, context_length=spec.CONTEXT_LENGTH,
                    token_dtype="<u4", tokenization="per_document_truncate_with_bos_no_packing",
                    active_tokens="all_document_tokens_except_initial_bos", splits={})
    for index, split in enumerate(("validation", "calibration", "train")):
        manifest["splits"][split] = dict(source_document_start=index, source_document_stop=index + 1,
                                         num_tokens=8)
    (tmp_path / spec.DATA_MANIFEST).write_text(json.dumps(manifest))
    cache = tmp_path / "activations"
    cache.mkdir()
    cached = dict(identity=cache_identity(manifest), shard_tokens=3, splits={})
    expected = (torch.arange(8 * spec.INPUT_DIM).reshape(8, spec.INPUT_DIM) / 100).bfloat16()
    for split in manifest["splits"]:
        (cache / split).mkdir()
        shards = []
        for start in range(0, len(expected), 3):
            stop = min(start + 3, len(expected))
            name = f"{split}/{start:09d}-{stop:09d}.bf16"
            content = expected[start:stop].contiguous().view(torch.uint16).numpy().tobytes()
            (cache / name).write_bytes(content)
            shards.append(dict(start=start, stop=stop, file=name,
                               sha256=hashlib.sha256(content).hexdigest()))
        cached["splits"][split] = dict(shards=shards, complete=True)
    (cache / spec.ACTIVATION_MANIFEST).write_text(json.dumps(cached))
    source = CachedGemmaActivationSource(tmp_path, device="cpu")
    torch.testing.assert_close(source.read(2, 5), expected[2:7].float(), rtol=0, atol=0)
    config = dict(path=str(tmp_path), training_tokens=8, buffer_tokens=4,
                  seed=17, normalization="none")
    stream = CachedActivationStream(config)
    permutation = np.random.default_rng(np.random.SeedSequence([17, 0])).permutation(4)
    oracle = expected[permutation].float()
    torch.testing.assert_close(stream.sample(3, 0), oracle[:3])
    torch.testing.assert_close(stream.sample(3, 0, "prior"), oracle[:3])
    torch.testing.assert_close(stream.sample(3, 0), oracle[:3])
    torch.testing.assert_close(stream.sample(3, 0, "validation"), expected[:3].float())
    source.close()


def test_synthetic_keyed_sampler_restores_rng():
    from befond.data.synthetic import sample_keyed_batch
    from befond.data.synthetic_spec import resolve_benchmark, MODEL_NUM_FEATURES

    class Features:
        feature_vectors = torch.ones(MODEL_NUM_FEATURES, 3)
        def __call__(self, codes):
            return codes @ self.feature_vectors

    generator = SimpleNamespace(feature_dict=Features(), activation_generator=SimpleNamespace(
        sample=lambda size: torch.rand(size, MODEL_NUM_FEATURES)))
    before = torch.random.get_rng_state().clone()
    x, codes = sample_keyed_batch(generator, 2, 9, 11)
    assert torch.equal(torch.random.get_rng_state(), before)
    repeated_x, repeated_codes = sample_keyed_batch(generator, 2, 9, 11)
    torch.testing.assert_close(x, repeated_x, rtol=0, atol=0)
    torch.testing.assert_close(codes, repeated_codes, rtol=0, atol=0)
    assert x.shape == (2, 3) and codes.shape == (2, MODEL_NUM_FEATURES)
    assert not torch.equal(codes, sample_keyed_batch(generator, 2, 300016, 11)[1])
    assert resolve_benchmark("historical").scale_children_by_parent is False
    assert resolve_benchmark("published").scale_children_by_parent is True
