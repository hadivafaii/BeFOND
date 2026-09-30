"""Evaluate a native checkpoint or pretrained bundle on held-out synthetic data."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from befond.checkpoints import load_checkpoint
from befond.config import load_config
from befond.data import make_source
from befond.evaluation.synthetic import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--config", help="Required for bundles that do not include training metadata.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--frequency-samples", type=int)
    args = parser.parse_args(argv)
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    config = load_config(args.config) if args.config else metadata.get("config")
    if config is None:
        parser.error("This checkpoint needs an explicit --config with data/evaluation settings.")
    config["model"] = asdict(model.config)
    bundled = getattr(model, "normalization_path", None)
    if bundled is not None:
        config["data"]["normalization_path"] = str(bundled)
    source = make_source(config, args.device)
    options = {k: v for k, v in config.get("evaluation", {}).items()
               if k in ("samples", "steps", "batch_size", "seed", "frequency_samples", "frequency_seed")}
    for name in ("samples", "batch_size", "steps", "frequency_samples"):
        if getattr(args, name) is not None:
            options[name] = getattr(args, name)
    options["beta"] = config["train"]["kl_beta"]
    metrics = evaluate(model, source, **options)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
