"""Fixed affine transforms fitted to training data, shared by training and evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class StreamingCovariance:
    """Accumulate population moments in FP64 without retaining observations."""

    def __init__(self, dimension, device="cpu", batch_reduction_dtype=torch.float32):
        self.batch_reduction_dtype = batch_reduction_dtype
        self.count = 0
        self.mean = torch.zeros(dimension, device=device, dtype=torch.float64)
        self.m2 = torch.zeros(dimension, dimension, device=device, dtype=torch.float64)

    def update(self, x):
        x = x.detach().flatten(1).to(device=self.mean.device, dtype=self.batch_reduction_dtype)
        n = len(x)
        batch_mean = x.mean(0)
        centered = x - batch_mean
        batch_m2 = centered.T @ centered
        mean = batch_mean.double()
        delta = mean - self.mean
        total = self.count + n
        self.m2.add_(batch_m2.double())
        self.m2.add_(torch.outer(delta, delta), alpha=self.count * n / total)
        self.mean.add_(delta, alpha=n / total)
        self.count = total

    def save(self, path, condition_cap=1e4, provenance=None):
        if self.count == 0:
            raise ValueError("Cannot normalize an empty data stream.")
        path = Path(path).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        mean = self.mean.cpu().numpy()
        covariance = (self.m2 / self.count).cpu().numpy()
        covariance = (covariance + covariance.T) / 2
        values, vectors = np.linalg.eigh(covariance)
        if values[-1] <= 0 or condition_cap <= 1:
            raise ValueError("Whitening requires nonzero variance and condition_cap > 1.")
        values = np.maximum(values, 0)
        regularized = np.maximum(values, values[-1] / condition_cap)
        whitener = (vectors / np.sqrt(regularized)) @ vectors.T
        dewhitener = (vectors * np.sqrt(regularized)) @ vectors.T
        arrays = dict(mean=mean, covariance=covariance,
                      whitener=((whitener + whitener.T) / 2).astype(np.float32),
                      dewhitener=((dewhitener + dewhitener.T) / 2).astype(np.float32),
                      scale=np.asarray(np.sqrt(np.trace(covariance) / len(mean)), dtype=np.float32),
                      global_mean=np.asarray(mean.mean()),
                      global_std=np.asarray(np.sqrt(np.trace(covariance) / len(mean)
                                                    + np.mean((mean - mean.mean()) ** 2))))
        files = {}
        for name, value in arrays.items():
            destination = path / f"{name}.npy"
            np.save(destination, value, allow_pickle=False)
            files[name] = dict(filename=destination.name,
                               sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
        manifest = dict(sample_count=self.count, dimension=len(mean),
                        condition_cap=condition_cap, batch_reduction_dtype=str(self.batch_reduction_dtype),
                        provenance=provenance or {}, files=files)
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return path


class ActivationNormalizer:
    """Use x'=(x-mean) @ whitener; dictionaries have shape [D,K]."""

    def __init__(self, mode="none", *, mean=None, scale=None, whitener=None,
                 dewhitener=None, manifest=None):
        if mode not in ("none", "whiten", "global_rms", "global_standardize", "center_global_rms"):
            raise ValueError(f"Unknown normalization: {mode}")
        self.mode, self.mean, self.scale = mode, mean, scale
        self.whitener, self.dewhitener, self.manifest = whitener, dewhitener, manifest
        self._cache = {}

    @classmethod
    def none(cls):
        return cls()

    @classmethod
    def from_artifact(cls, mode, path):
        if mode == "none":
            return cls.none()
        path = Path(path).expanduser()
        if path.is_file():
            result = cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            if result.mode != mode:
                raise ValueError("Saved normalization mode does not match the configuration.")
            return result
        manifest = json.loads((path / "manifest.json").read_text())
        arrays = {}
        for name, record in manifest["files"].items():
            file = path / record["filename"]
            if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Normalization checksum mismatch: {file}")
            arrays[name] = np.load(file, allow_pickle=False)
        mean = arrays["mean"]
        if mode == "whiten":
            return cls(mode, mean=mean, whitener=arrays["whitener"],
                       dewhitener=arrays["dewhitener"], manifest=manifest)
        if mode == "global_rms":
            scale = np.sqrt(np.trace(arrays["covariance"]) / len(mean) + np.mean(mean ** 2))
            mean = np.zeros_like(mean)
        elif mode == "global_standardize":
            scale = arrays["global_std"]
            mean = np.full_like(mean, arrays["global_mean"])
        else:
            scale = arrays["scale"]
        return cls(mode, mean=mean, scale=float(scale), manifest=manifest)

    def _parameters(self, like):
        key = (like.device, like.dtype)
        if key not in self._cache:
            convert = lambda x: torch.as_tensor(x, device=like.device, dtype=like.dtype)
            self._cache[key] = (convert(self.mean),
                                convert(self.whitener) if self.mode == "whiten" else None,
                                convert(self.dewhitener) if self.mode == "whiten" else None)
        return self._cache[key]

    def transform(self, x):
        if self.mode == "none":
            return x
        mean, whitener, _ = self._parameters(x)
        return (x - mean) @ whitener if self.mode == "whiten" else (x - mean) / self.scale

    def inverse(self, x):
        if self.mode == "none":
            return x
        mean, _, dewhitener = self._parameters(x)
        return x @ dewhitener + mean if self.mode == "whiten" else x * self.scale + mean

    def decoder_to_raw(self, decoder):
        if self.mode == "none":
            return decoder
        _, _, dewhitener = self._parameters(decoder)
        return dewhitener.T @ decoder if self.mode == "whiten" else decoder * self.scale

    def bias_to_raw(self, bias):
        return self.inverse(bias)

    def state_dict(self):
        result = {"mode": self.mode, "manifest": self.manifest}
        for name in ("mean", "scale", "whitener", "dewhitener"):
            value = getattr(self, name)
            result[name] = None if value is None else torch.as_tensor(value)
        return result

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)
