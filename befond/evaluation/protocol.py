"""Small protocol metadata helpers shared by the optional benchmark adapters."""

from dataclasses import asdict
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from befond.data.gemma_spec import SAEBENCH_COMMIT


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def task_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}/{name}".encode()).digest()[:4], "little")


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _output_fields(config, adapter, variant_id):
    return dict(eval_config=config, eval_id=str(uuid.uuid4()),
                datetime_epoch_millis=int(time.time() * 1000),
                sae_bench_commit_hash=SAEBENCH_COMMIT, sae_lens_id=variant_id,
                sae_lens_release_id="befond", sae_lens_version=version("sae-lens"),
                sae_cfg_dict=adapter.cfg.to_dict())
