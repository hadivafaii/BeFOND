"""Fit fixed ZCA/scale transforms on the full prepared Gemma training stream."""

import argparse
import json
from pathlib import Path

import torch

from befond.data.activation_cache import CachedGemmaActivationSource
from befond.data.normalization import StreamingCovariance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/gemma")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--condition-cap", type=float, default=1e4)
    args = parser.parse_args(argv)
    source = CachedGemmaActivationSource(args.data_dir, device=args.device)
    total = source.manifest["splits"]["train"]["num_tokens"]
    if source.num_tokens != total:
        raise ValueError("Complete training activation extraction before fitting normalization.")
    destination = Path(args.data_dir).expanduser() / "normalization"
    if (destination / "manifest.json").exists():
        raise FileExistsError(f"Normalization already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    progress = destination / "progress.pt"
    moments = StreamingCovariance(source.input_dim, args.device, batch_reduction_dtype=torch.float64)
    identity = source.cache_manifest["identity"]
    if progress.exists():
        state = torch.load(progress, map_location=args.device, weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Saved normalization progress belongs to different data.")
        moments.count, moments.mean, moments.m2 = state["count"], state["mean"], state["m2"]
    for start in range(moments.count, total, args.batch_size):
        moments.update(source.read(start, min(args.batch_size, total - start)))
        if (start // args.batch_size) % 100 == 0 or moments.count == total:
            temporary = progress.with_suffix(".tmp")
            torch.save(dict(count=moments.count, mean=moments.mean, m2=moments.m2,
                            identity=identity), temporary)
            temporary.replace(progress)
            print(f"Normalization: {moments.count:,}/{total:,} tokens", flush=True)
    moments.save(destination, args.condition_cap, provenance={"source": identity})
    progress.unlink(missing_ok=True)
    source.close()
    print(f"Normalization saved to {destination}")


if __name__ == "__main__":
    main()
