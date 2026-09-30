"""Generate and cache superposition dictionaries using the recorded generator recipe."""

import argparse
import json
from pathlib import Path
import tempfile
import time

from befond.data.files import sha256_file
from befond.data.synthetic_spec import SNAPSHOT_FILES, verify_synthsaebench_snapshot
import befond.data.synthetic_spec as spec


def generator_config(dimension):
	config = json.loads(Path(spec.__file__).with_name('test1_generator_config.json').read_text())
	config['hidden_dim'] = dimension
	return config


def prepare(root, device):
	import sae_lens
	import torch
	from sae_lens.synthetic import SyntheticModel, SyntheticModelConfig

	if sae_lens.__version__ != '6.49.1':
		raise ValueError('superposition generation requires sae-lens==6.49.1.')
	root = Path(root).expanduser().resolve()
	root.mkdir(parents=True, exist_ok=True)
	dimensions = (1536, 1024, 768, 512, 384, 256)
	for dimension in dimensions:
		destination = root / f'd{dimension}'
		if destination.exists():
			verify_synthsaebench_snapshot(destination, hidden_dim=dimension)
			print(f'd={dimension}: reusing verified dictionary.', flush=True)
			continue
		start = time.monotonic()
		config = generator_config(dimension)
		with tempfile.TemporaryDirectory(dir=root) as temporary:
			staging = Path(temporary) / 'generator'
			model = SyntheticModel(SyntheticModelConfig.from_dict(config), device=device)
			model.save(staging)
			record = dict(config=config, sae_lens=sae_lens.__version__, torch=torch.__version__,
				device=str(device), files={name: sha256_file(staging / name) for name in SNAPSHOT_FILES})
			(staging / 'test1_generation.json').write_text(json.dumps(record, indent=2) + '\n')
			verify_synthsaebench_snapshot(staging, hidden_dim=dimension)
			staging.rename(destination)
			del model
		print(f'd={dimension}: generated in {time.monotonic() - start:.1f}s.', flush=True)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--device', default='cuda:0')
	args = parser.parse_args()
	prepare(args.output, args.device)


if __name__ == '__main__':
	main()
