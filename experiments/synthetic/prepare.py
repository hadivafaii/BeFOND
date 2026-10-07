"""Download the pinned generator and fit normalization on a separate stream."""

import argparse
import json
from pathlib import Path

from befond.config import config_path
from befond.data.normalization import StreamingCovariance
from befond.data.synthetic import load_synthsaebench, sample_keyed_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthetic.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    config = json.loads(config_path(args.config).read_text())
    data = config["data"]
    generator = load_synthsaebench(data["benchmark"], snapshot=data.get("generator_path"),
                                  hidden_dim=config["model"]["input_dim"], device=args.device)
    if data["normalization"] == "none":
        return
    samples = args.samples or data.get("normalization_samples", 2000000000)
    batch_size = args.batch_size or data.get("normalization_batch_size", 50000)
    destination = Path(data["normalization_path"]).expanduser()
    if (destination / "manifest.json").exists():
        raise FileExistsError(f"Normalization already exists: {destination}")
    moments = StreamingCovariance(config["model"]["input_dim"], args.device)
    for step, start in enumerate(range(0, samples, batch_size)):
        x, _ = sample_keyed_batch(generator, min(batch_size, samples - start),
                                  data.get("normalization_seed", 4000037), step)
        moments.update(x)
        if step % 100 == 0:
            print(f"Normalization: {moments.count:,}/{samples:,} examples", flush=True)
    moments.save(destination, provenance={"benchmark": data["benchmark"],
                                          "seed": data.get("normalization_seed", 4000037)})
    print(f"Normalization saved to {destination}")


if __name__ == "__main__":
    main()
