"""Portable, width-independent model bundles and resumable training checkpoints.

A published bundle contains config.json, model.pt, metadata.json and, when
needed, normalization.pt. Loading never imports the research repository or
executes code downloaded from the Hub. Model tensors use weights-only loading.
"""

import argparse
import json
import os
import shutil
from dataclasses import asdict, fields
from pathlib import Path

import torch

from .config import json_ready, write_config
from .model import BeFOND, ModelConfig


SCHEMA_VERSION = 1


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(model, path, *, config, step, source_state=None):
    """Save the completed update count and all state required by the trainer."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_config": json_ready(asdict(model.config)),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": json_ready(config),
        "step": int(step),
        "torch_rng_state": torch.get_rng_state(),
        "source_state": source_state,
    }
    if model.decoder.device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(model.decoder.device).cpu()
    _atomic_save(payload, path)


def load_checkpoint(path, device="cpu"):
    """Return ``(model, training_state)`` from a native checkpoint or bundle."""
    path = Path(path).expanduser()
    if path.is_dir():
        model = from_pretrained(path, device=device)
        return model, model.pretrained_metadata
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema; use the legacy converter for research fits")
    # Older saved configs without this field used the original ETD1 default.
    payload["model_config"].setdefault("inference_integrator", "etd1")
    model = BeFOND(ModelConfig(**payload["model_config"]))
    # Respect precision stored in a checkpoint (including float64 references).
    model.to(dtype=payload["model"]["decoder"].dtype)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), {k: v for k, v in payload.items() if k != "model"}


def save_pretrained(model, directory, *, metadata=None, normalization_path=None):
    """Export one model width, ready for a subfolder in a future Hub repository."""
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    write_config(directory / "config.json", asdict(model.config))
    _atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                 directory / "model.pt")
    info = dict(metadata or {})
    info.update(schema_version=SCHEMA_VERSION, architecture="BeFOND",
                input_dim=model.config.input_dim, num_latents=model.config.num_latents)
    if normalization_path is not None:
        source = Path(normalization_path).expanduser()
        destination = directory / "normalization.pt"
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        info["normalization_file"] = "normalization.pt"
    write_config(directory / "metadata.json", info)
    return directory


def from_pretrained(path_or_repo, *, subfolder=None, revision=None,
                    device="cpu", local_files_only=False):
    """Load a local bundle or explicit Hub repo/subfolder/revision.

    No repository name or release list is assumed: checkpoints can be published
    independently for every width. Remote loading downloads only selected files.
    The returned model carries ``normalization_path`` and ``pretrained_metadata``;
    experiment adapters apply the saved transform to raw activations.
    """
    local = Path(path_or_repo).expanduser()
    if local.is_dir():
        directory = local / subfolder if subfolder else local

        def fetch(name):
            return directory / name
    else:
        from huggingface_hub import hf_hub_download

        def fetch(name):
            return Path(hf_hub_download(
                repo_id=str(path_or_repo), filename=name, subfolder=subfolder,
                revision=revision, local_files_only=local_files_only))

    with fetch("config.json").open() as handle:
        config = json.load(handle)
    config.setdefault("inference_integrator", "etd1")
    with fetch("metadata.json").open() as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported pretrained bundle schema")
    state = torch.load(fetch("model.pt"), map_location="cpu", weights_only=True)
    model = BeFOND(ModelConfig(**config)).to(dtype=state["decoder"].dtype)
    model.load_state_dict(state, strict=True)
    model.pretrained_metadata = metadata
    model.normalization_path = (
        fetch(metadata["normalization_file"]) if metadata.get("normalization_file") else None)
    return model.to(device).eval()


def convert_legacy(checkpoint, model_config, output, *, normalization_path=None,
                   normalization_mode="whiten", training_config=None, trusted=False):
    """Convert a research Bernoulli NGD checkpoint without importing its code.

    ``trusted=True`` explicitly opts into pickle loading for older private fits.
    Public bundles always load with weights_only=True.
    """
    with Path(model_config).expanduser().open() as handle:
        original = json.load(handle)
    if original.get("truncation_ceiling") != 1 or original.get("learning_mode", "ngd") != "ngd":
        raise ValueError("Only Bernoulli NGD research checkpoints can be converted")
    if original.get("latent_pixels", 1) != 1:
        raise ValueError("Conversion requires a non-spatial dictionary")
    if original.get("inference_mode", "exact") != "exact" or original.get("stochastic", False):
        raise ValueError("Conversion requires deterministic exact Bernoulli inference")
    if original.get("clamp_u") is not None or original.get("clamp_du") is not None:
        raise ValueError("State-clamped research inference is not supported by this converter")
    if original.get("clamp_dec") is not None:
        raise ValueError("Convert effective variance explicitly for clamped research models")
    payload = torch.load(Path(checkpoint).expanduser(), map_location="cpu",
                         weights_only=not trusted)
    source = payload["model"]
    decoder = source["dec.weight"].flatten(1).T.contiguous()
    names = {field.name for field in fields(ModelConfig)}
    config = {k: v for k, v in original.items() if k in names}
    config.setdefault("inference_integrator", "etd1")
    config.update(input_dim=decoder.shape[0], num_latents=decoder.shape[1])
    model = BeFOND(ModelConfig(**config)).to(dtype=decoder.dtype)
    state = model.state_dict()
    state["decoder"] = decoder
    state["prior_logits"] = source["u_init.0"].reshape_as(state["prior_logits"])
    # Development checkpoints store log standard deviation, not log variance.
    state["log_variance"] = (2 * source["dec_log_sigma"]).reshape_as(state["log_variance"])
    if "bias" in state:
        state["bias"] = source["dec.bias"].reshape_as(state["bias"])
    model.load_state_dict(state, strict=True)
    provenance_keys = ("global_step", "train_steps", "created_timestamp", "seed")
    metadata = {"converted_from": Path(checkpoint).name,
                "research_metadata": json_ready({key: value for key, value in
                    payload.get("metadata", {}).items() if key in provenance_keys})}
    if training_config:
        from .config import TrainConfig
        settings = json.loads(Path(training_config).read_text())
        names = {field.name for field in fields(TrainConfig)}
        metadata["train"] = {key: value for key, value in settings.items() if key in names}
    if normalization_path is not None and Path(normalization_path).expanduser().is_dir():
        from .data.normalization import ActivationNormalizer
        normalizer = ActivationNormalizer.from_artifact(normalization_mode, normalization_path)
        normalization_path = Path(output).expanduser() / "normalization.pt"
        _atomic_save(normalizer.state_dict(), normalization_path)
    return save_pretrained(model, output, metadata=metadata,
                           normalization_path=normalization_path)


def main():
    parser = argparse.ArgumentParser(description="Convert a research Bernoulli NGD fit to a portable bundle")
    parser.add_argument("checkpoint")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalization")
    parser.add_argument("--normalization-mode", default="whiten",
                        choices=("whiten", "global_rms", "global_standardize", "center_global_rms"),
                        help="Mode when converting a research normalization directory")
    parser.add_argument("--trusted", action="store_true", help="Allow pickle in a trusted local research checkpoint")
    args = parser.parse_args()
    directory = convert_legacy(args.checkpoint, args.model_config, args.output,
                               normalization_path=args.normalization,
                               normalization_mode=args.normalization_mode,
                               training_config=args.training_config, trusted=args.trusted)
    print(directory)


if __name__ == "__main__":
    main()
