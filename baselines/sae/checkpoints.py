"""Portable SAE weights and explicit architecture configuration."""
from dataclasses import asdict
import json
from pathlib import Path

from .config import SAEConfig
from .models import build_sae


def save_sae(model, directory):
    """Write config.json and model.safetensors; no optimizer or executable pickle."""
    from safetensors.torch import save_file

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text(json.dumps(asdict(model.config), indent=2) + '\n')
    weights = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    temporary = path / 'model.safetensors.tmp'
    save_file(weights, temporary)
    temporary.replace(path / 'model.safetensors')


def load_sae(source, *, device='cpu', subfolder='', revision=None, local_files_only=False):
    """Load a local directory or a Hugging Face repo once weights are published."""
    from safetensors.torch import load_file

    path = Path(source)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        prefix = f'{subfolder.strip("/")}/' if subfolder else ''
        path = Path(snapshot_download(str(source), revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=[prefix + 'config.json', prefix + 'model.safetensors']))
    path = path / subfolder
    model = build_sae(SAEConfig(**json.loads((path / 'config.json').read_text())))
    model.load_state_dict(load_file(path / 'model.safetensors'))
    return model.to(device).eval()
