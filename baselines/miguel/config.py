"""All current MF/GAMP defaults and benchmark presets (no accelerator imports)."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    method: str = "mf"
    target: str = "saebench"
    width: int = 16384
    dim: int = 2304
    seed: int = 0
    steps: int = 20
    damping: float = .2
    first_damping: float = 1.
    w_lr: float = .0003
    moment_lr: float = .01
    tokens: int = 49_999_872
    batch: int = 2048
    buffer: int = 16384
    init_backend: str = "cpu"
    norm_max: float = 4.
    normalization: str = "none"
    prior: float = .01
    clip_ratio: float = 0.
    skip_ratio: float = 0.
    guard_warmup: int = 0
    max_skip_rate: float = 1.


def selected_config(method, width, seed, target='saebench', training_tokens=None, whiten=False):
    """One reproduction recipe per target; no hyperparameter search."""
    if whiten:
        if method != 'mf' or target != 'saebench' or training_tokens is not None \
                or width not in (16384, 65536, 131072, 524288) or seed not in range(5):
            raise ValueError('Whitened SAEBench is MF, 50M tokens, widths 16384–524288, model seeds 0–4')
        return Config(width=width, seed=seed, init_backend='cpu' if width == 16384 else 'gpu',
                      w_lr=.0005, moment_lr=.01 if width == 524288 else .003, norm_max=1.58,
                      normalization='whiten', prior=.01*16384/width, clip_ratio=3., skip_ratio=5.,
                      guard_warmup=2000, max_skip_rate=.2)
    widths = {'saebench': (16384, 65536), 'synthsaebench': (16384,)}
    if method not in ('mf', 'gamp') or target not in widths or width not in widths[target] or seed not in range(5):
        raise ValueError('Choose a supported benchmark/width and model seed 0–4')
    if target == 'synthsaebench':
        if training_tokens is not None:
            raise ValueError('--training-tokens is only for SAEBench')
        w, moment = (.001, .003) if method == 'mf' else (.03, .03)
        return Config(method=method, target=target, width=width, dim=768,
                      seed=seed, init_backend='gpu', batch=1024, buffer=8192,
                      tokens=199_999_488, damping=.2 if method == 'mf' else .5,
                      first_damping=1. if method == 'mf' else .5,
                      w_lr=w, moment_lr=moment)
    budget = 50_000_000 if training_tokens is None else training_tokens
    if budget not in (50_000_000, 500_000_000):
        raise ValueError('SAEBench training_tokens must be 50000000 or 500000000')
    rate = .001 if width == 16384 else .003
    w_rate = .0003
    if budget == 500_000_000:
        rate = .003 if width == 16384 else .001
        w_rate = .0003 if width == 16384 else .0001
    return Config(method=method, width=width, seed=seed,
                  tokens=budget//2048*2048,
                  init_backend='cpu' if width == 16384 else 'gpu',
                  damping=.2 if method == 'mf' else .5,
                  first_damping=1. if method == 'mf' else .5,
                  w_lr=w_rate if method == 'mf' else rate,
                  moment_lr=.01 if method == 'mf' else rate)

