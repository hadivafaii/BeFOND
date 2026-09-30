"""Training hyperparameters and JSON configuration loading.

The experiment JSON files are the first place to look for the actual settings.
These defaults mirror the general NGD defaults of the development repository.
Unknown fields are errors: an override must never silently do nothing.
"""

import copy
import json
from dataclasses import MISSING, asdict, dataclass, fields
from importlib.metadata import distribution
from pathlib import Path


@dataclass
class TrainConfig:
    lr: float = 1.0
    lr_min: float = 1e-5
    batch_size: int = 5000
    train_steps: int = 12000
    stop_after_updates: int | None = None
    warmup_portion: float = 0.0
    warm_restart: int = 0
    warm_restart_peak_mult: float = 1.0
    scheduler_type: str = "cos"
    kl_beta: float = 1.0
    horizon_dist: str = "exp-10"

    ngd_delta_t: float = 1.0
    ngd_decoder_metric: str = "prior"
    ngd_decoder_solver: str = "dense"
    ngd_fisher_damping: float = 0.0
    ngd_moment_mode: str = "current"
    ngd_chebyshev_degree: int = 128
    ngd_phi1_auto_taylor_max_applications: int = 32
    ngd_phi1_auto_chebyshev_tolerance: float = 1e-5
    ngd_phi1_auto_chebyshev_max_degree: int = 512
    ngd_lr_bias_mult: float = 1.0
    ngd_lr_sigma_mult: float = 1.0
    ngd_lr_prior_mult: float = 1.0
    ngd_freeze_variance_after: int | None = None
    ngd_freeze_variance_scope: str = "all"

    ngd_prior_update_every: int = 500
    ngd_prior_t_outer: int | str = "auto"
    ngd_prior_t_outer_max: int = 400
    ngd_prior_kl_window: int = 10
    ngd_prior_kl_threshold: float | list[float] = 0.01
    ngd_prior_kl_reject_threshold: float = 0.1
    ngd_prior_batch_size: int | None = None
    ngd_prior_n_batches: int = 1
    r0_budget: float | None = 0.01
    r0_floor: float = 0.0

    decoder_norm_method: str = "none"
    initial_decoder_norm: float | None = None
    decoder_norm_target: float | list[float] = 1.0
    decoder_norm_lamb_start: float = 10.0
    decoder_norm_lamb_end: float = 10.0
    decoder_norm_lamb_anneal_portion: float = 0.0

    eval_freq: int = 1000
    eval_batch_size: int = 4096
    eval_n_batches: int = 1
    eval_final_n_samples: int | None = None
    chkpt_freq: int = 6000
    log_freq: int = 10

    def __post_init__(self):
        for key in ("batch_size", "train_steps", "ngd_prior_update_every",
                    "ngd_prior_t_outer_max", "ngd_prior_kl_window", "ngd_prior_n_batches",
                    "eval_batch_size", "eval_n_batches", "log_freq"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.eval_freq < 0 or self.chkpt_freq < 0:
            raise ValueError("eval_freq and chkpt_freq must be nonnegative (0 disables)")
        if self.ngd_moment_mode != "current":
            raise ValueError("BeFOND uses the current Bernoulli posterior moments")
        if self.kl_beta < 0 or self.ngd_delta_t <= 0:
            raise ValueError("kl_beta must be nonnegative and ngd_delta_t positive")
        if not 0 <= self.r0_floor < 1:
            raise ValueError("r0_floor must lie in [0, 1)")
        if self.r0_budget is not None and not self.r0_floor <= self.r0_budget < 1:
            raise ValueError("r0_budget must lie in [r0_floor, 1)")
        if self.ngd_prior_t_outer != "auto" and int(self.ngd_prior_t_outer) < 1:
            raise ValueError("ngd_prior_t_outer must be 'auto' or a positive integer")
        if self.ngd_freeze_variance_scope not in ("all", "multistep"):
            raise ValueError("ngd_freeze_variance_scope must be all or multistep")
        if self.decoder_norm_method not in ("none", "clip", "soft"):
            raise ValueError("decoder_norm_method must be none, clip, or soft")


def json_ready(value):
    """Represent infinity explicitly rather than writing nonstandard JSON."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    return value


def load_config(path, overrides=()):
    """Read JSON and apply explicit ``section.key=JSON`` command-line overrides."""
    with config_path(path).open() as handle:
        config = json.load(handle)
    from .model import ModelConfig
    # Populate visible dataclass defaults before validating override names.
    # Leave derived train_steps=None unresolved until token-budget overrides land.
    for section, cls in (("model", ModelConfig), ("train", TrainConfig)):
        defaults = {field.name: field.default for field in fields(cls) if field.default is not MISSING}
        config[section] = {**defaults, **config.get(section, {})}
    config["logging"] = {"wandb": False, "project": "BeFOND", "entity": None,
                         "name": None, **config.get("logging", {})}
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator or "." not in key:
            raise ValueError(f"Expected section.key=JSON, got {override!r}")
        section, name = key.split(".", 1)
        if section not in config or name not in config[section]:
            raise ValueError(f"Unknown configuration key: {key}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        config[section][name] = value
    return resolve_config(config)


def config_path(path):
    """Find a config in a checkout or the installed distribution's data files."""
    path = Path(path).expanduser()
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    checkout = Path(__file__).resolve().parents[1] / path
    if checkout.is_file():
        return checkout
    # pip --target places data-files under the target rather than sys.prefix.
    target = Path(__file__).resolve().parents[1] / "share" / "befond" / path
    if target.is_file():
        return target
    package = distribution("befond")
    for entry in package.files or ():
        if str(entry).endswith(f"share/befond/configs/{path.name}"):
            installed = Path(package.locate_file(entry))
            if installed.is_file():
                return installed
    raise FileNotFoundError(path)


def resolve_config(config):
    from .model import ModelConfig

    result = copy.deepcopy(config)
    allowed = {"model", "train", "data", "evaluation", "logging", "description", "provenance"}
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    result["model"] = asdict(ModelConfig(**result["model"]))
    train = result.get("train", {})
    if train.get("train_steps", 1) is None:
        train["train_steps"] = result["data"]["training_tokens"] // train["batch_size"]
    result["train"] = asdict(TrainConfig(**train))
    return result


def write_config(path, config):
    Path(path).write_text(json.dumps(json_ready(config), indent=2, allow_nan=False) + "\n")
