"""Write a portable experiment plan, or execute one row on the selected device."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .load import list_runs


def make_plan(suite):
    runs = list_runs(headline_only=suite == 'headline')
    if suite == 'screen':
        runs = [r for r in runs if r['stage'] in ('r1', 'r2', 'r3')]
    return {'schema_version': 1, 'suite': suite, 'runs': runs}


def command_for(row, *, out_root, device='cuda', offline=False):
    cfg = row['config']
    name = cfg['recipe']['name']
    # Preserve ad-hoc period overrides and their original folder suffixes.
    stem = row['run_id'].split('/')[1].rsplit('_k', 1)[0]
    overrides = {k: v for k, v in cfg['recipe'].items() if k != 'name'}
    cmd = [sys.executable, '-m', 'baselines.david.train',
           '--recipe', name, '--k', str(cfg['k']), '--seed', str(cfg['seed']),
           '--samples', str(cfg['samples']), '--batch', str(cfg['batch']),
           '--width', str(cfg['width']), '--lr', str(cfg['lr']),
           '--eval-samples', str(row['eval_samples']), '--tag', row['stage'],
           '--suffix', stem[len(name):], '--override', json.dumps(overrides),
           '--out-root', str(out_root), '--device', device]
    if offline:
        cmd.append('--offline')
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['headline', 'screen', 'all'], default='headline')
    parser.add_argument('--output', type=Path, default=Path('width_sweep_plan.json'))
    parser.add_argument('--run', type=Path, help='execute one row from this plan')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--out-root', type=Path, default=Path('outputs/david'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.run:
        runs = json.loads(args.run.read_text())['runs']
        if not 0 <= args.index < len(runs):
            parser.error(f'index must be between 0 and {len(runs) - 1}')
        command = command_for(runs[args.index], out_root=args.out_root, device=args.device, offline=args.offline)
        print(json.dumps({'run_id': runs[args.index]['run_id'], 'command': command}), flush=True)
        subprocess.run(command, check=True)
    else:
        plan = make_plan(args.suite)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2) + '\n')
        print(f"Wrote {len(plan['runs'])} runs to {args.output}")


if __name__ == '__main__':
    main()
