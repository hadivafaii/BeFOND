"""Train any small SAE on a BeFOND data source using explicit JSON settings."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import SAEConfig
from .training import TrainingConfig, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from befond.data import make_source
    from .models import build_sae

    raw = json.loads(args.config.read_text())
    model_cfg = SAEConfig(**raw['model'])
    train_cfg = TrainingConfig(**raw.get('train', {}))
    resolved = dict(model=asdict(model_cfg), train=asdict(train_cfg), data=raw['data'])
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'run.json'
    if args.resume:
        if json.loads(config_path.read_text()) != json.loads(json.dumps(resolved)):
            raise ValueError('Resume requires the same model, training and data configuration')
    else:
        if config_path.exists():
            raise FileExistsError(f'{config_path}: use --resume or a new output directory')
        config_path.write_text(json.dumps(resolved, indent=2) + '\n')
    model = build_sae(model_cfg).to(args.device)
    source_config = dict(resolved, model=dict(resolved['model'], input_dim=model_cfg.input_dim))
    source = make_source(source_config, args.device)
    train(model, source, train_cfg, output=args.output, resume=args.resume)


if __name__ == '__main__':
    main()
