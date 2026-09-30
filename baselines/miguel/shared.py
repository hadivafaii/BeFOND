"""Shared configuration, data, checkpointing and evaluation for MF/GAMP.

The numerical method implementations live in mf.py and gamp.py.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

# The Torch sampler/metric peer runs in its own Python environment. It never
# imports JAX, the learner, or model initialization.
if len(sys.argv) < 2 or sys.argv[1] != '_synth-peer':
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import ml_dtypes


from .config import Config, selected_config


def project(p, norm_min=0., norm_max=4.):
    result = dict(p)
    norms = jnp.maximum(jnp.linalg.norm(p['w'], axis=1), 1e-12)
    result['w'] = p['w']*(jnp.clip(norms, norm_min, norm_max)/norms)[:, None]
    return result


def initialize(cfg):
    # Keep the two norm operations separate and use the configured backend.
    # Fusing this initializer changes floating-point rounding.
    if jax.config.jax_enable_x64:
        raise ValueError('Set JAX_ENABLE_X64=false for FP32 initialization')
    with jax.default_device(jax.devices(cfg.init_backend)[0]):
        w = jax.random.normal(jax.random.key(cfg.seed), (cfg.width, cfg.dim))
        w *= 1. / jnp.linalg.norm(w, axis=1, keepdims=True)
        p = dict(w=w, b=jnp.zeros(cfg.dim),
                 logits=jnp.full(cfg.width, jnp.log(.01/(1-.01))),
                 logvar=jnp.log(jnp.asarray(1.)))
        p['logvar'] = jnp.full((cfg.dim,), p['logvar'])
        return project(p, 3., 4.)


def finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(tree)]))


def make_chunk(cfg, method, mesh):
    """Compile eight-update chunks with GAMP's per-step shard_map reduction."""
    if cfg.method == 'gamp':
        def local(p, s, x):
            p, s, aux = method.update(p, s, x, cfg, 'data')
            return p, s, jax.lax.pmean(aux, 'data')
        step = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P(), P(), P('data', None)),
                            out_specs=(P(), P(), P()), check_vma=False))
    else:
        step = jax.jit(lambda p, s, x: method.update(p, s, x, cfg))

    def chunk(p, s, xs):
        def one(c, x):
            p, s, a = step(*c, x)
            if cfg.target == 'saebench':
                a = (a['parameters_finite'] > 0) & finite(s)
            return (p, s), a
        (p, s), a = jax.lax.scan(one, (p, s), xs)
        ok = jnp.all(a['parameters_finite']) & finite(s) if cfg.target == 'synthsaebench' else jnp.all(a)
        return p, s, ok, a
    return jax.jit(chunk)


def array_digest(p):
    return hashlib.sha256(b''.join(np.asarray(p[k]).tobytes() for k in sorted(p))).hexdigest()



def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def whitening(directory):
    """Load row-vector transforms: z=(x-mean)@whitener; x=z@dewhitener+mean."""
    if directory is None:
        raise ValueError('Whitened configurations require --normalization')
    return [jnp.asarray(np.load(Path(directory)/f'{name}.npy'), dtype=jnp.float32)
            for name in ('mean', 'whitener', 'dewhitener')]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


class ActivationCache:
    """Read the shared raw Gemma activation cache before optional whitening."""
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'activation_manifest.json').read_text())
        identity = self.manifest['identity']
        required = dict(hf_model_name='google/gemma-2-2b',
            model_revision='c5ebcd40d208330abc697524c919956e692655cf',
            hook_name='blocks.12.hook_resid_post', input_dim=2304,
            storage_dtype='bfloat16', normalization='none')
        for k, v in required.items():
            if identity[k] != v:
                raise ValueError(f'Cache identity mismatch: {k}')

    def read(self, split, start, count):
        shards = self.manifest['splits'][split]['shards']
        if start<0 or count<1 or start+count>shards[-1]['stop']:
            raise ValueError('Missing cache interval')
        out = np.empty((count, 2304), dtype=ml_dtypes.bfloat16)
        done = 0
        while done<count:
            shard = shards[start//self.manifest['shard_tokens']]
            n = min(count-done, shard['stop']-start)
            view = memoryview(out[done:done+n].view(np.uint8)).cast('B')
            with (self.root/shard['file']).open('rb', buffering=0) as f:
                f.seek((start-shard['start'])*2304*2)
                got = 0
                while got<len(view):
                    size = f.readinto(view[got:])
                    if not size:
                        raise EOFError(shard['file'])
                    got += size
            start += n
            done += n
        return out


def buffers(cache, cfg, start=0):
    rng = np.random.default_rng(cfg.seed)
    def jobs():
        for offset in range(0, cfg.tokens, cfg.buffer):
            n = min(cfg.buffer, cfg.tokens-offset)
            order = rng.permutation(n).astype(np.int32)
            if offset>=start:
                yield offset, n, order
    schedule_iter = iter(jobs())
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = deque()
        def enqueue():
            item = next(schedule_iter, None)
            if item is not None:
                offset, n, order = item
                pending.append((offset, order, pool.submit(cache.read, 'train', offset, n)))
        for _ in range(4):
            enqueue()
        while pending:
            offset, order, future = pending.popleft()
            raw = future.result()
            enqueue()
            yield offset, raw, order


def prepare_buffer(mesh, batch):
    raw_s, rep, out = NamedSharding(mesh, P('data', None)), NamedSharding(mesh, P()), NamedSharding(mesh, P(None, 'data', None))
    prepare = jax.jit(lambda raw, order: raw[order].astype(jnp.float32).reshape((-1, batch, raw.shape[-1])),
                      in_shardings=(raw_s, rep), out_shardings=out)
    return lambda raw, order: prepare(jax.device_put(raw, raw_s), jax.device_put(order, rep))


def save_checkpoint(path, p, state, cfg):
    leaves, _ = jax.tree.flatten(state)
    payload = {f'params_{k}': np.asarray(v) for k, v in jax.device_get(p).items()}
    payload.update({f'optimizer_{i}': np.asarray(v) for i, v in enumerate(jax.device_get(leaves))})
    payload['metadata'] = np.asarray(json.dumps(dict(config=asdict(cfg), optimizer_leaves=len(leaves))))
    path = Path(path)
    temporary = path.with_suffix('.partial.npz')
    np.savez(temporary, **payload)
    temporary.replace(path)


def restore(path, template, cfg):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        if meta['config'] != asdict(cfg):
            raise ValueError('Resume would change the experiment or schedule')
        leaves, tree = jax.tree.flatten(template)
        if len(leaves) != meta['optimizer_leaves']:
            raise ValueError('Optimizer structure changed')
        values = [z[f'optimizer_{i}'] for i in range(len(leaves))]
        for value, leaf in zip(values, leaves, strict=True):
            if value.shape != leaf.shape or value.dtype != leaf.dtype:
                raise ValueError('Optimizer shape/dtype changed')
        p = {k.removeprefix('params_'): z[k] for k in z.files if k.startswith('params_')}
        if set(p) != {'w', 'b', 'logvar', 'logits'} or p['w'].shape != (cfg.width, cfg.dim):
            raise ValueError('Checkpoint parameter structure changed')
        if any(not np.isfinite(v).all() for v in (*p.values(), *values)):
            raise ValueError('Nonfinite checkpoint')
        return p, jax.tree.unflatten(tree, values)


def train(args, method):
    cfg = Config(**json.loads(args.config.read_text()))
    if len(jax.devices()) != 1:
        raise ValueError('Expose one JAX device per training run')
    if cfg.target not in ('saebench', 'synthsaebench'):
        raise ValueError('Unknown benchmark')
    if cfg.method != method.NAME or cfg.tokens%cfg.batch or cfg.buffer != 8*cfg.batch:
        raise ValueError('Method/batch mismatch')
    versions = {k: importlib.metadata.version(k) for k in ('jax', 'jaxlib', 'numpy', 'optax')}
    if cfg.target == 'saebench' and args.cache is None:
        raise ValueError('SAEBench training requires --cache')
    if cfg.target == 'synthsaebench' and (cfg.dim != 768 or cfg.batch != 1024 or args.sampler_python is None or args.snapshot is None):
        raise ValueError('Synthetic training requires D=768, B=1024, --snapshot and --sampler-python')
    cache = ActivationCache(args.cache) if cfg.target == 'saebench' else None
    norm = whitening(args.normalization) if cfg.normalization == 'whiten' else None
    if args.checkpoint_every is not None and (args.checkpoint_every < 1 or args.checkpoint_every%cfg.buffer):
        raise ValueError('Checkpoints must be at complete shuffle-buffer boundaries')
    p = initialize(cfg)
    initial_sha = array_digest(p)
    if cfg.prior != .01:
        # Width-scaled initial prior; the checksum above covers the default initializer.
        p = dict(p, logits=jnp.full_like(p['logits'], float(jnp.log(cfg.prior/(1-cfg.prior)))))
    state = method.initial_state(p, cfg)
    run = args.output
    if args.resume:
        if json.loads((run/'config.json').read_text()) != asdict(cfg):
            raise ValueError('Resume output does not belong to this configuration')
        p, state = restore(args.resume, state, cfg)
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run/'config.json', asdict(cfg))
    done = int(state[1])*cfg.batch
    if done%cfg.buffer:
        raise ValueError('Resume must be at a complete shuffle-buffer boundary')
    stop = cfg.tokens if args.stop_tokens is None else args.stop_tokens
    if not done<stop<=cfg.tokens or (stop != cfg.tokens and stop%cfg.buffer):
        raise ValueError('Invalid stopping point; schedule horizon is never shortened')
    write_json(run/'run_manifest.json', dict(initial_parameters_sha256=initial_sha,
        cache_manifest_sha256=file_digest(args.cache/'activation_manifest.json') if cache else None,
        data=cache.manifest if cache else dict(benchmark='historical-parent-unscaled', data_seed=1,
            generator_revision=SYNTH_REVISION, batch=1024, split='trn', shuffle=False), versions=versions,
        source_sha256={n: file_digest(Path(__file__).with_name(n)) for n in ('mf.py', 'gamp.py', 'shared.py', 'config.py')},
        devices=[str(d) for d in jax.devices()], config=asdict(cfg)))
    mesh = Mesh(np.array(jax.devices()), ('data',))
    p, state = jax.device_put((p, state), NamedSharding(mesh, P()))
    chunk = make_chunk(cfg, method, mesh)
    prepare = prepare_buffer(mesh, cfg.batch)
    if norm:
        mean, whitener, _ = norm
        whiten = jax.jit(lambda xs: jnp.matmul(xs-mean, whitener, precision='highest'))
    begin = time.perf_counter()
    segment_start, digest = done, hashlib.sha256()
    inputs = buffers(cache, cfg, done) if cache else synthetic_buffers(args, cfg, done, stop)
    with (run/'history.jsonl').open('a') as history:
        for offset, raw, order in inputs:
            digest.update(raw.tobytes())
            xs = prepare(raw, order) if cache else jax.device_put(
                raw.reshape(-1, cfg.batch, cfg.dim), NamedSharding(mesh, P()))
            if norm:
                xs = whiten(xs)
            p, state, ok, aux = jax.block_until_ready(chunk(p, state, xs))
            if not bool(ok):
                raise FloatingPointError(f'Nonfinite state at {offset}')
            done = offset+len(raw)
            if offset == 0 or done//1_000_000 != offset//1_000_000 or done == stop:
                row = dict(tokens=done, seconds=time.perf_counter()-begin)
                history.write(json.dumps(row)+'\n'); history.flush()
                print(json.dumps(row), flush=True)
            if args.checkpoint_every and done%args.checkpoint_every == 0 and done < cfg.tokens:
                # Parameter-only checkpoint; its directory is a valid --run for evaluation.
                partial = run/'checkpoints'/f'{done:09d}'
                partial.mkdir(parents=True)
                write_json(partial/'config.json', asdict(cfg))
                np.savez(partial/'checkpoint.npz', **jax.device_get(p))
            if done == stop and cache:
                inputs.close()
                break
    save_checkpoint(run/f'state-{done}.npz', p, state, cfg)
    np.savez(run/'checkpoint.npz', **jax.device_get(p))
    write_json(run/'summary.json', dict(tokens=done, completed=done==cfg.tokens,
        parameters_sha256=array_digest(p), wall_seconds=time.perf_counter()-begin,
        segment_start=segment_start, input_segment_sha256=digest.hexdigest()))


SYNTH_REVISION = 'b2efd8b919ae46d6d487c73d46db5ee52813621d'


def read_exact(pipe, size):
    blocks = []
    while size:
        block = pipe.read(size)
        if not block:
            raise EOFError('Synthetic peer ended before the requested batch')
        blocks.append(block)
        size -= len(block)
    return b''.join(blocks)


@contextlib.contextmanager
def synthetic_peer(args, mode, log, extra=()):
    command = [str(args.sampler_python), '-u', '-m', f'{__package__}.shared',
               '_synth-peer', '--mode', mode, '--snapshot', str(args.snapshot), *extra]
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
        env.pop(key, None)
    with Path(log).open('x') as error:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=error)
        try:
            yield proc
            proc.stdin.close()
            if proc.stdout.read(1) or proc.wait(timeout=180) != 0:
                raise RuntimeError(f'Synthetic peer failed; see {log}')
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)


def synthetic_buffers(args, cfg, start, stop):
    with synthetic_peer(args, 'stream', args.output/f'producer-{start}-{stop}.log',
                        ('--start', str(start//cfg.batch), '--batches', str((stop-start)//cfg.batch))) as proc:
        for offset in range(start, stop, cfg.buffer):
            n = min(cfg.buffer, stop-offset)
            raw = read_exact(proc.stdout, n*cfg.dim*4)
            yield offset, np.frombuffer(raw, dtype='<f4').reshape(n, cfg.dim), None


def synthetic_evaluate(args, method):
    cfg = Config(**json.loads((args.run/'config.json').read_text()))
    if cfg.target != 'synthsaebench' or cfg.method != method.NAME:
        raise ValueError('Wrong benchmark or method')
    if args.output.exists() or not 0 < args.samples <= 500_000:
        raise ValueError('Use a new output path and 1–500000 evaluation samples')
    with np.load(args.run/'checkpoint.npz', allow_pickle=False) as z:
        p = {k: jnp.asarray(z[k]) for k in z.files}
    encode = jax.jit(lambda p, x: method.probabilities(p, x, cfg))
    decode = jax.jit(lambda r, w, b: r@w+b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with synthetic_peer(args, 'score', args.output.with_suffix('.peer.log'),
                        ('--run', str(args.run), '--samples', str(args.samples), '--output', str(args.output),
                         *(['--rare'] if args.rare else []))) as proc:
        for _ in range((args.samples+1023)//1024):
            x = jnp.asarray(np.frombuffer(read_exact(proc.stdout, 1024*768*4), dtype='<f4').reshape(1024, 768))
            r = jax.block_until_ready(encode(p, x))
            reconstruction = decode(r, p['w'], p['b'])
            proc.stdin.write(np.asarray(r, dtype='<f4').tobytes())
            proc.stdin.write(np.asarray(reconstruction, dtype='<f4').tobytes())
            proc.stdin.flush()


def synthetic_peer_main():
    """Torch-only process: public generator, keyed sampling, official metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'stream', 'score'), required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--batches', type=int, default=0)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--samples', type=int, default=500_000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rare', action='store_true')
    args = parser.parse_args(sys.argv[2:])
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from befond.data.synthetic import load_synthsaebench, sample_keyed_batch
        from befond.data.synthetic_spec import download_snapshot, resolve_benchmark, MODEL_REVISION
        if MODEL_REVISION != SYNTH_REVISION:
            raise ValueError('Wrong generator revision')
        if args.mode == 'prepare':
            download_snapshot(resolve_benchmark('synthsaebench16k-historical-fond-unscaled'), args.snapshot)
            return
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        model = load_synthsaebench('synthsaebench16k-historical-fond-unscaled', snapshot=args.snapshot, device='cuda:0')
        def sample(step, split):
            offset = {'trn': 0, 'vld': 100_003, 'tst': 200_003, 'prior': 300_007}[split]
            x, g = sample_keyed_batch(model, 1024, 1+offset, step, autocast=False)
            return x.reshape(1024, 768), g.reshape(1024, 16384)
    if args.mode == 'stream':
        if args.start < 0 or args.batches < 1:
            raise ValueError('Invalid stream range')
        for step in range(args.start, args.start+args.batches):
            x, _ = sample(step, 'trn')
            sys.stdout.buffer.write(x.cpu().contiguous().numpy().tobytes())
        sys.stdout.buffer.flush()
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.run/'checkpoint.npz', allow_pickle=False) as z:
        dictionary = torch.from_numpy(z['w'].copy()).cuda()
    scores = SyntheticMetrics(dictionary, model.feature_dict.feature_vectors)
    rare = None
    if args.rare:
        from befond.evaluation.feature_metrics import GroundTruthFeatureMetrics, feature_frequency_reference
        frequencies = feature_frequency_reference(model)
        frequency_hash = hashlib.sha256(frequencies.numpy().tobytes()).hexdigest()
        rare = GroundTruthFeatureMetrics(dictionary, model.feature_dict.feature_vectors, frequencies)
    digest = hashlib.sha256()
    begin = time.perf_counter()
    with torch.no_grad():
        for step in range((args.samples+1023)//1024):
            x, truth = sample(step, 'tst')
            raw = x.cpu().contiguous().numpy().tobytes()
            digest.update(raw)
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            r = np.frombuffer(read_exact(sys.stdin.buffer, 1024*len(dictionary)*4), dtype='<f4').reshape(1024, len(dictionary))
            reconstruction = np.frombuffer(read_exact(sys.stdin.buffer, 1024*768*4), dtype='<f4').reshape(1024, 768)
            n = min(1024, args.samples-step*1024)
            scores.add_batch(torch.from_numpy(r[:n].copy()).cuda(), truth[:n], x[:n],
                             torch.from_numpy(reconstruction[:n].copy()).cuda())
            if rare is not None:
                rare.add_batch(torch.from_numpy((r[:n] > .5).astype(np.float32)).cuda(), truth[:n])
    result = scores.compute()
    if rare is not None:
        result.update(feature_recovery=rare.compute(), frequency_reference_sha256=frequency_hash,
                      frequency_reference=dict(samples=500000, batch=1024, seed=315031))
    write_json(args.output, result | dict(observation_stream_sha256=digest.hexdigest(),
        checkpoint_sha256=file_digest(args.run/'checkpoint.npz'), split='tst', data_seed=1,
        batch=1024, sample_identity_passed=True, elapsed_seconds=time.perf_counter()-begin))




class SyntheticMetrics:
    def __init__(self, dictionary, ground_truth):
        import torch
        from scipy.optimize import linear_sum_assignment
        from sae_lens.util import cosine_similarities
        assert dictionary.ndim == ground_truth.ndim == 2
        assert dictionary.shape[1] == ground_truth.shape[1]
        assert dictionary.shape[0] > 0 and ground_truth.shape[0] > 0
        assert dictionary.dtype == ground_truth.dtype == torch.float32
        assert torch.isfinite(dictionary).all() and torch.isfinite(ground_truth).all()
        begin = time.perf_counter()
        similarity = cosine_similarities(dictionary.detach(), ground_truth.detach()).abs()
        assert torch.isfinite(similarity).all()
        # Same full cost matrix and solver as the pinned official calculator.
        row, column = linear_sum_assignment(1 - similarity.cpu().numpy())
        self.mcc = similarity[row, column].mean().item()
        self.matches = similarity.argmax(dim=1)
        self.uniqueness = self.matches.unique().numel() / dictionary.shape[0]
        self.alignment_seconds = time.perf_counter() - begin
        self.width = dictionary.shape[0]
        self.dimension = dictionary.shape[1]
        self.truth_width = ground_truth.shape[0]
        self.tp, self.fp, self.fn, self.tn = [
            torch.zeros(self.width, dtype=torch.int64, device=dictionary.device)
            for _ in range(4)]
        self.samples = 0
        self.active = 0
        self.true_active = 0
        self.expected_active = 0.0
        self.squared_error = 0.0

    def add_batch(self, probabilities, truth_coefficients, observations, reconstruction):
        import torch
        n = len(probabilities)
        assert n > 0 and probabilities.shape == (n, self.width)
        assert truth_coefficients.shape == (n, self.truth_width)
        assert observations.shape == reconstruction.shape == (n, self.dimension)
        assert torch.isfinite(probabilities).all() and torch.isfinite(truth_coefficients).all()
        assert torch.isfinite(observations).all() and torch.isfinite(reconstruction).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        fires = probabilities > .5
        truth = truth_coefficients[:, self.matches] > 0
        self.tp += (fires & truth).sum(dim=0)
        self.fp += (fires & ~truth).sum(dim=0)
        self.fn += (~fires & truth).sum(dim=0)
        self.tn += (~fires & ~truth).sum(dim=0)
        self.samples += n
        self.active += fires.sum().item()
        self.true_active += (truth_coefficients > 0).sum().item()
        self.expected_active += probabilities.sum(dtype=torch.float64).item()
        self.squared_error += (observations - reconstruction).square().sum(dtype=torch.float64).item()

    def compute(self):
        import torch
        assert self.samples > 0
        # Official SAE-Lens reduces the per-feature ratios on CPU in FP32.
        # Counts below 500000 are represented exactly in FP32.
        tp, fp, fn, tn = [x.cpu().float() for x in (self.tp, self.fp, self.fn, self.tn)]
        precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
        recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
        f1 = torch.where(precision + recall > 0,
            2 * precision * recall / (precision + recall), torch.zeros_like(tp))
        accuracy = (tp + tn) / (tp + fp + fn + tn).clamp(min=1e-8)
        return dict(samples=self.samples, mcc=self.mcc, uniqueness=self.uniqueness,
            precision=precision.mean().item(), recall=recall.mean().item(),
            f1_score=f1.mean().item(), accuracy=accuracy.mean().item(),
            natural_l0=self.active / self.samples,
            expected_l0=self.expected_active / self.samples,
            true_l0=self.true_active / self.samples,
            never_firing=int((tp + fp == 0).sum().item()),
            mse_per_coordinate=self.squared_error / (self.samples * self.dimension),
            alignment_seconds=self.alignment_seconds,
            readout='natural Bernoulli probability > 0.5; posterior-mean reconstruction',
            matching='MCC: one-to-one absolute cosine; F1: per-learned-atom maximum absolute cosine')



MODEL_REVISION = 'c5ebcd40d208330abc697524c919956e692655cf'
SAEBENCH_COMMIT = '8042bb3828c6340da8d12062324e92b2077c571c'


def evaluation_runtime(need_model=False):
    # The official pre/post-balancing annotations reuse one dimension name.
    # Disable those runtime annotations, not numerical benchmark code.
    import jaxtyping
    import torch
    jaxtyping.config.update('jaxtyping_disable', True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if len(jax.devices()) != 1 or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one GPU for evaluation')
    # Only load caches you generated yourself. Remap producer CUDA ordinals
    # to the visible GPU without changing values or dtypes.
    torch.serialization.register_package(0, lambda storage: None,
        lambda storage, location: storage.cuda(0) if location.startswith('cuda') else None)
    if need_model:
        torch.set_num_threads(2)
        from transformer_lens import HookedTransformer
        return torch, HookedTransformer
    return torch, None


def sae_probes(args, method):
    """Official 113-task probes, sharing BeFOND's public activation-cache workflow."""
    from functools import lru_cache
    from sae_bench.evals.sparse_probing_sae_probes.eval_config import SparseProbingSaeProbesEvalConfig
    from befond.evaluation.sae_probes import run_sae_probes
    torch, loader = evaluation_runtime(need_model=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = SparseProbingSaeProbesEvalConfig(model_name='gemma-2-2b', random_seed=42)
    adapter = make_adapter(args.run, method, normalization=args.normalization)

    @lru_cache(maxsize=1)
    def get_llm():
        model = loader.from_pretrained_no_processing('gemma-2-2b', device='cuda',
            dtype=torch.bfloat16, revision=MODEL_REVISION)
        return model.eval().requires_grad_(False)

    result = run_sae_probes(config, adapter, get_llm, args.cache, args.output, args.run.name)
    write_json(args.output/'sae_probes.json', result)
    write_json(args.output/'manifest.json', dict(config=asdict(config), saebench_commit=SAEBENCH_COMMIT,
        model_revision=MODEL_REVISION, checkpoint_sha256=file_digest(args.run/'checkpoint.npz')))


def make_adapter(run, method, decode_like_codes=False, normalization=None):
    """Native inference, unit-row decoder interface; differentiable decode."""
    import torch
    from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig
    cfg = Config(**json.loads((run/'config.json').read_text()))
    if cfg.method != method.NAME or cfg.target != 'saebench':
        raise ValueError('Wrong method entry point')
    norm = whitening(normalization) if cfg.normalization == 'whiten' else None
    with np.load(run/'checkpoint.npz', allow_pickle=False) as z:
        params = {k: jnp.asarray(z[k]) for k in z.files}

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if norm is None:
                w = torch.from_numpy(np.asarray(params['w']).copy()).cuda()
                self.W_dec = torch.nn.Parameter(w/w.norm(dim=1)[:, None], requires_grad=False)
                self.b_dec = torch.nn.Parameter(torch.from_numpy(np.asarray(params['b']).copy()).cuda(), requires_grad=False)
                self.W_enc = torch.nn.Parameter(torch.zeros_like(w.T), requires_grad=False)
                self.b_enc = torch.nn.Parameter(torch.zeros(len(w), device='cuda'), requires_grad=False)
                self._encode = jax.jit(lambda p, x: (method.probabilities(p, x, cfg)>.5).astype(x.dtype)*jnp.linalg.norm(p['w'], axis=1))
            else:
                # Raw-coordinate interface: whiten inputs, dewhiten atoms and bias;
                # binary codes are scaled by the raw-coordinate atom norms.
                mean, whitener, dewhitener = norm
                raw = np.asarray(jnp.matmul(params['w'], dewhitener, precision='highest'))
                norms = np.linalg.norm(raw, axis=1)
                bias = np.asarray(jnp.matmul(params['b'], dewhitener, precision='highest')+mean)
                self.W_dec = torch.nn.Parameter(torch.from_numpy((raw/norms[:, None]).copy()).cuda(), requires_grad=False)
                self.b_dec = torch.nn.Parameter(torch.from_numpy(bias.copy()).cuda(), requires_grad=False)
                self.W_enc = torch.nn.Parameter(torch.zeros((cfg.dim, cfg.width), device='cuda'), requires_grad=False)
                self.b_enc = torch.nn.Parameter(torch.zeros(cfg.width, device='cuda'), requires_grad=False)
                scale = jnp.asarray(norms)
                self._encode = jax.jit(lambda p, x: (method.probabilities(
                    p, jnp.matmul(x-mean, whitener, precision='highest'), cfg)>.5).astype(x.dtype)*scale)
            self.cfg = CustomSAEConfig(model_name='gemma-2-2b', hook_name='blocks.12.hook_resid_post',
                hook_layer=12, d_in=cfg.dim, d_sae=cfg.width)
            self.cfg.architecture = 'binary_iterative_'+method.NAME
            self.cfg.normalize_activations = 'none'
            self.cfg.device, self.cfg.dtype = 'cuda:0', 'float32'
            self.cfg.training_tokens, self.cfg.context_size = cfg.tokens, 1024
            self.cfg.dataset_path = 'monology/pile-uncopyrighted'

        @property
        def device(self):
            return self.W_dec.device

        @property
        def dtype(self):
            return self.W_dec.dtype

        def to(self, *args, **kwargs):
            super().to(*args, **kwargs)
            if self.dtype != torch.float32 or self.device.type != 'cuda':
                raise ValueError('Adapter parameters must remain CUDA FP32')
            self.cfg.device = str(self.device)
            return self

        def encode(self, x):
            flat = x.detach().reshape(-1, cfg.dim).to(self.device, torch.float32).contiguous()
            output = []
            for start in range(0, len(flat), 1024):
                chunk = flat[start:start+1024]
                n = len(chunk)
                if n<1024:
                    chunk = torch.nn.functional.pad(chunk, (0, 0, 0, 1024-n))
                value = self._encode(params, jax.dlpack.from_dlpack(chunk))
                output.append(torch.utils.dlpack.from_dlpack(value)[:n])
            return torch.cat(output).reshape(*x.shape[:-1], cfg.width)

        def decode(self, codes):
            result = codes.to(device=self.device, dtype=self.dtype)@self.W_dec+self.b_dec
            return result.to(codes.dtype) if decode_like_codes else result

        def forward(self, x):
            return self.decode(self.encode(x))

    return Adapter()


def ravel(args, method):
    import random
    from types import SimpleNamespace
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from befond.evaluation.ravel import RAVEL_DATASET_REVISION
    torch, _ = evaluation_runtime()
    from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
    from sae_bench.evals.ravel import main as upstream, mdbm
    args.output.mkdir(parents=True, exist_ok=False)
    config = RAVELEvalConfig(model_name='google/gemma-2-2b', llm_batch_size=32,
                            artifact_dir=str(args.cache))
    # Match the official intervention's tensor/tuple and BF16 return boundary;
    # SAE inference and the differentiable decoder matmul remain FP32.
    original = mdbm.MDBM.create_intervention_hook
    original_tokenizer = upstream.AutoTokenizer
    def pinned_tokenizer(name, **kwargs):
        return AutoTokenizer.from_pretrained(name, **(kwargs | dict(revision=MODEL_REVISION)))
    def compatible_hook(self, *a, **kw):
        hook = original(self, *a, **kw)
        def call(module, inputs, outputs):
            if isinstance(outputs, torch.Tensor):
                return hook(module, inputs, (outputs,))[0]
            return hook(module, inputs, outputs)
        return call
    mdbm.MDBM.create_intervention_hook = compatible_hook
    upstream.AutoTokenizer = SimpleNamespace(from_pretrained=pinned_tokenizer)
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map='cuda',
            torch_dtype=torch.bfloat16, attn_implementation='eager', revision=MODEL_REVISION)
        model.eval().requires_grad_(False)
        tokenizer = pinned_tokenizer(config.model_name)
        # Build both model-only datasets before SAE scoring. The official
        # scorer then resets its RNG, so cold/warm preparation cannot consume
        # a different amount of randomness during mask fitting.
        args.cache.mkdir(parents=True, exist_ok=True)
        snapshot_download('adamkarvonen/ravel_prompts', repo_type='dataset',
            revision=RAVEL_DATASET_REVISION, local_dir=str(args.cache/'base'), allow_patterns='*.json')
        cache_hashes = {}
        for entity, attributes in config.entity_attribute_selection.items():
            path = args.cache/upstream.get_instance_name(entity, config.model_name,
                config.full_dataset_downsample, config.top_n_entities)
            if not path.exists():
                random.seed(config.random_seed)
                torch.manual_seed(config.random_seed)
                np.random.seed(config.random_seed)
                original_batch = config.llm_batch_size
                try:
                    config.llm_batch_size *= 8
                    with torch.no_grad():
                        dataset = upstream.RAVELInstance.create_from_files(config=config,
                            entity_type=entity, tokenizer=tokenizer, data_dir=config.artifact_dir,
                            model=model, model_name=config.model_name, attribute_types=attributes,
                            downsample=config.full_dataset_downsample)
                finally:
                    config.llm_batch_size = original_batch
                dataset.create_and_save_filtered_dataset(artifact_dir=config.artifact_dir,
                    top_n_entities=config.top_n_entities)
            cache_hashes[path.name] = file_digest(path)
        adapter = make_adapter(args.run, method, decode_like_codes=True, normalization=args.normalization)
        begin = time.perf_counter()
        result, details = upstream.run_eval_single_sae(config, adapter, model, 'cuda', str(args.output/'artifacts'))
        write_json(args.output/'ravel.json', dict(metrics=result, details=details,
            eval_config=asdict(config), wall_seconds=time.perf_counter()-begin,
            checkpoint_sha256=file_digest(args.run/'checkpoint.npz'), saebench_commit=SAEBENCH_COMMIT,
            model_revision=MODEL_REVISION, dataset_revision=RAVEL_DATASET_REVISION,
            model_cache_sha256=cache_hashes))
    finally:
        mdbm.MDBM.create_intervention_hook = original
        upstream.AutoTokenizer = original_tokenizer


def main(method):
    """Run a preset, a training job, or an evaluation without a private checkout."""
    parser = argparse.ArgumentParser(description=method.__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    config = sub.add_parser('config', help='Write an explicit benchmark configuration')
    config.add_argument('--target', choices=('saebench', 'synthsaebench'), default='saebench')
    config.add_argument('--training-tokens', type=int, choices=(50_000_000, 500_000_000))
    config.add_argument('--whiten', action='store_true')
    config.add_argument('--width', type=int, default=16384)
    config.add_argument('--seed', type=int, default=0)
    config.add_argument('--output', type=Path, required=True)
    training = sub.add_parser('train')
    training.add_argument('--config', type=Path, required=True)
    training.add_argument('--cache', type=Path, help='Raw Gemma activation cache')
    training.add_argument('--sampler-python', type=Path, help='Python in the synthetic data environment')
    training.add_argument('--snapshot', type=Path, help='Local SynthSAEBench world')
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--resume', type=Path)
    training.add_argument('--stop-tokens', type=int)
    training.add_argument('--checkpoint-every', type=int)
    training.add_argument('--normalization', type=Path, help='Directory of whitening .npy arrays')
    for name in ('sae-probes', 'ravel'):
        evaluation = sub.add_parser(name)
        evaluation.add_argument('--run', type=Path, required=True)
        evaluation.add_argument('--cache', type=Path, required=True)
        evaluation.add_argument('--output', type=Path, required=True)
        evaluation.add_argument('--normalization', type=Path)
    synthetic = sub.add_parser('synthetic-eval')
    synthetic.add_argument('--run', type=Path, required=True)
    synthetic.add_argument('--sampler-python', type=Path, required=True)
    synthetic.add_argument('--snapshot', type=Path, required=True)
    synthetic.add_argument('--samples', type=int, default=500_000)
    synthetic.add_argument('--rare', action='store_true')
    synthetic.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'config':
        cfg = selected_config(method.NAME, args.width, args.seed, args.target,
                              args.training_tokens, args.whiten)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, asdict(cfg))
        print(json.dumps(asdict(cfg), indent=2))
    elif args.command == 'train':
        train(args, method)
    elif args.command == 'sae-probes':
        sae_probes(args, method)
    elif args.command == 'ravel':
        ravel(args, method)
    elif args.command == 'synthetic-eval':
        synthetic_evaluate(args, method)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_synth-peer':
        synthetic_peer_main()
    else:
        raise SystemExit('Use python -m baselines.miguel.mf (or baselines.miguel.gamp)')
