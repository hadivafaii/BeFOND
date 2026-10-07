"""Run current SAE Probes and RAVEL protocols for a frozen BeFOND checkpoint."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from befond.checkpoints import load_checkpoint
from befond.config import load_config, json_ready
from befond.data.normalization import ActivationNormalizer
from befond.evaluation.adapter import BeFONDAdapter
from befond.evaluation.protocol import _save_json


def configure_inference(model, evaluation):
    """Apply recorded evaluation overrides without changing training settings."""
    fields = ("t_inner", "inference_h", "inference_prior_retention")
    overrides = {key: evaluation[key] for key in fields if key in evaluation}
    model.config = replace(model.config, **overrides)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="data/gemma/evaluation")
    parser.add_argument("--evals", nargs="+", choices=("sae_probes", "ravel"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--sae-probes-datasets", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    config = load_config(args.config) if args.config else metadata.get("config")
    if config is None:
        parser.error("This checkpoint needs an explicit --config with normalization/evaluation settings.")
    evaluation = config["evaluation"]
    configure_inference(model, evaluation)
    normalization_path = (getattr(model, "normalization_path", None)
                          or config["data"].get("normalization_path"))
    normalizer = ActivationNormalizer.from_artifact(config["data"]["normalization"], normalization_path)
    adapter = BeFONDAdapter(model, normalizer,
                            steps=args.steps or evaluation["steps"],
                            beta=config["train"]["kl_beta"],
                            microbatch_size=args.microbatch_size or evaluation.get("microbatch_size", 256),
                            readout=evaluation.get("readout", "posterior_threshold"),
                            input_scale=config["data"].get("input_scale", 1.0))
    settings = SimpleNamespace(seed=evaluation.get("seed", 42),
                               ravel_batch_size=evaluation.get("ravel_batch_size", 32),
                               sae_probes_datasets=args.sae_probes_datasets)
    output, cache = Path(args.output), Path(args.cache)
    output.mkdir(parents=True, exist_ok=True)
    # Separate output directories identify checkpoint, normalization, and inference choices.
    identity = dict(checkpoint=str(Path(args.checkpoint).resolve()), config=config,
                    steps=adapter.steps, readout=adapter.readout,
                    microbatch_size=adapter.microbatch_size, model_config=asdict(model.config),
                    normalization_path=str(normalization_path))
    identity = json_ready(identity)
    import json
    identity_path = output / "evaluation_config.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Output directory belongs to another evaluation configuration.")
    _save_json(identity_path, identity)
    for family in args.evals or evaluation.get("evals", ["sae_probes", "ravel"]):
        result_path = output / f"{family}.json"
        if result_path.exists() and not args.force:
            print(f"Reusing {result_path}")
            continue
        if family == "sae_probes":
            from befond.data.gemma import load_gemma
            from befond.evaluation.sae_probes import sae_probes_config, run_sae_probes
            models = []
            def get_model():
                if not models:
                    models.append(load_gemma(args.device))
                return models[0]
            result = run_sae_probes(sae_probes_config(settings), adapter, get_model,
                                    cache / family, output / family, "befond", force=args.force)
            del models
        else:
            from befond.evaluation.ravel import (
                ravel_config, ravel_tasks, run_ravel, merge_ravel, load_ravel_model)
            protocol = ravel_config(settings)
            llm = load_ravel_model(args.device)
            results = {}
            for entity, task in ravel_tasks(protocol).items():
                entity_path = output / family / f"{entity}.json"
                if entity_path.exists() and not args.force:
                    results[entity] = json.loads(entity_path.read_text())
                else:
                    results[entity] = run_ravel(task, adapter, llm, cache / family, "befond")
                    _save_json(entity_path, results[entity])
            result = merge_ravel(protocol, results)
            del llm
        _save_json(result_path, result)
        print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
