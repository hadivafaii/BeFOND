"""Load one selected paper SAE from the published checkpoint release."""
from pathlib import Path, PurePosixPath
import json

REPO_ID = 'chanind-goodfire/synthsaebench-full-width-sweep'
REVISION = '13630f6252a41d01e645f5c6714d39cafdb65f00'


def list_runs():
    """Return the six selected original/modified synthetic paper recipes."""
    return json.loads(Path(__file__).with_name('catalog.json').read_text())['runs']


def load_sae(run_id, *, device='cpu', revision=REVISION, local_files_only=False):
    """Download cfg.json and safetensors for a catalog run, then load with SAELens.

    Inputs are unnormalized 768-dimensional samples from the released synthetic
    world. Activation normalization is already folded into the inference weights.
    """
    from huggingface_hub import snapshot_download
    from sae_lens import SAE

    path = PurePosixPath(run_id)
    if path.is_absolute() or '..' in path.parts or run_id not in {r['run_id'] for r in list_runs()}:
        raise ValueError(f'Unknown run_id: {run_id!r}; see list_runs()')
    subdir = f'runs/{run_id}/final'
    snapshot = snapshot_download(
        REPO_ID, revision=revision, local_files_only=local_files_only,
        allow_patterns=[f'{subdir}/cfg.json', f'{subdir}/sae_weights.safetensors'],
    )
    return SAE.load_from_disk(Path(snapshot) / subdir, device=device).eval()
