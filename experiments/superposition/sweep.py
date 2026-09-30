"""Write explicit configurations for the current superposition search grid."""

import argparse
import copy
import itertools
import json
import math
from pathlib import Path

from befond.config import config_path

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/superposition.json")
    parser.add_argument("--grid", default="configs/superposition_sweep.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--generator-root", default="data/superposition")
    args = parser.parse_args(argv)
    base = json.loads(config_path(args.base).read_text())
    grid = json.loads(config_path(args.grid).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for dimension, lr, beta, depth in itertools.product(
            grid["dimensions"], grid["learning_rates"], grid["betas"], grid["depths"]):
        config = copy.deepcopy(base)
        config["model"].update(input_dim=dimension,
                               init_scale=grid["initial_component_scale_numerator"] / math.sqrt(dimension),
                               t_outer=depth["t_outer"])
        config["train"].update(lr=lr, kl_beta=beta, horizon_dist=depth["horizon_dist"],
                               initial_decoder_norm=grid["initial_decoder_norms"][str(dimension)])
        config["data"]["generator_path"] = str(Path(args.generator_root) / f"d{dimension}")
        name = f"d{dimension}_lr{lr}_beta{beta}_{depth['horizon_dist']}.json"
        (output / name).write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(list(output.glob('*.json')))} experiment configurations to {output}")


if __name__ == "__main__":
    main()
