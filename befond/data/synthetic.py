"""Official SynthSAEBench loading and deterministic BeFOND data streams."""

import contextlib
import pathlib
import random

import numpy as np
import torch

from .synthetic_spec import (
	HIDDEN_DIM,
	MODEL_NUM_FEATURES,
	BenchmarkSpec,
	ensure_synthsaebench_snapshot,
	resolve_benchmark,
	verify_synthsaebench_snapshot,
)

_BATCH_SEED_STRIDE = 1_000_003
_SEED_MODULUS = 2 ** 63 - 1
_STREAM_SEED_OFFSETS = {
	"trn": 0,
	"vld": 100_003,
	"tst": 200_003,
	"prior": 300_007,
}


def _hierarchy_nodes(roots):
	stack = list(roots)
	while stack:
		node = stack.pop()
		yield node
		stack.extend(node.children)


def _set_parent_scaling(synthetic_model, enabled):
	hierarchy = synthetic_model.hierarchy
	if hierarchy is None:
		raise ValueError("The pinned SynthSAEBench model has no hierarchy.")

	try:
		from sae_lens.synthetic import ActivationGenerator, hierarchy_modifier
	except ModuleNotFoundError as error:
		raise ModuleNotFoundError(
			"SynthSAEBench requires the optional sae_lens package."
		) from error

	for node in _hierarchy_nodes(hierarchy.roots):
		node.scale_children_by_parent = bool(enabled and node.children)
	hierarchy.modifier = hierarchy_modifier(hierarchy.roots)
	if synthetic_model.cfg.hierarchy is not None:
		synthetic_model.cfg.hierarchy.scale_children_by_parent = bool(enabled)

	# ActivationGenerator wraps the hierarchy modifier at construction time.
	# Rebuild it through its public constructor while retaining every sampled
	# distribution parameter from the pinned snapshot.
	old_generator = synthetic_model.activation_generator
	synthetic_model.activation_generator = ActivationGenerator(
		num_features=old_generator.num_features,
		firing_probabilities=old_generator.firing_probabilities,
		std_firing_magnitudes=old_generator.std_firing_magnitudes,
		mean_firing_magnitudes=old_generator.mean_firing_magnitudes,
		modify_activations=hierarchy.modifier,
		correlation_matrix=synthetic_model.correlation_matrix,
		device=old_generator.firing_probabilities.device,
		dtype=old_generator.firing_probabilities.dtype,
		use_sparse_tensors=old_generator.use_sparse_tensors,
	)
	return synthetic_model


def load_synthsaebench(
		benchmark: str | BenchmarkSpec,
		*,
		snapshot=None,
		hidden_dim=HIDDEN_DIM,
		device="cpu", ):
	"""Load a local or Hub snapshot and apply an explicit benchmark identity.

	``benchmark`` is intentionally required: the published and historical BeFOND
	distributions differ in how hierarchy magnitudes are scaled.
	"""

	spec = resolve_benchmark(benchmark)
	try:
		import sae_lens.synthetic.synthetic_model as synthetic_model_module
		from sae_lens.synthetic import SyntheticModel
	except ModuleNotFoundError as error:
		raise ModuleNotFoundError(
			"SynthSAEBench requires the optional sae_lens package."
		) from error

	snapshot = (
		spec.generator_dir
		if snapshot is None else pathlib.Path(snapshot).expanduser()
	)
	if hidden_dim != HIDDEN_DIM and snapshot == spec.generator_dir:
		raise ValueError("Test 1 requires its saved dimension-specific generator snapshot.")
	if snapshot == spec.generator_dir:
		snapshot = ensure_synthsaebench_snapshot(spec)
	else:
		snapshot = verify_synthsaebench_snapshot(snapshot, hidden_dim=hidden_dim)
	# SAE-Lens 6.49.1 otherwise snapshots every visible CUDA RNG here.
	original_temporary_seed = synthetic_model_module.temporary_seed
	synthetic_model_module.temporary_seed = lambda seed: (
		contextlib.nullcontext()
		if seed is None else _isolated_rng(seed, device)
	)
	try:
		model = SyntheticModel.load_from_disk(snapshot, device=str(device))
	finally:
		synthetic_model_module.temporary_seed = original_temporary_seed
	return _set_parent_scaling(
		model, enabled=spec.scale_children_by_parent)


def _batch_seed(data_seed, step):
	if step < 0:
		raise ValueError("step must be non-negative")
	return int((int(data_seed) + _BATCH_SEED_STRIDE * int(step)) % _SEED_MODULUS)


@contextlib.contextmanager
def _isolated_rng(seed, device):
	"""Seed upstream global samplers, then restore every touched RNG."""

	python_state = random.getstate()
	numpy_state = np.random.get_state()
	device = torch.device(device)
	cuda_devices = []
	if device.type == "cuda":
		cuda_devices = [
			device.index
			if device.index is not None else torch.cuda.current_device()
		]
	mps_state = None
	if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
		mps_state = torch.mps.get_rng_state()

	try:
		with torch.random.fork_rng(devices=cuda_devices):
			random.seed(seed)
			np.random.seed(seed % (2 ** 32))
			torch.random.default_generator.manual_seed(seed)
			if device.type == "cuda":
				with torch.cuda.device(device):
					torch.cuda.manual_seed(seed)
			elif device.type == "mps" and hasattr(torch.mps, "manual_seed"):
				torch.mps.manual_seed(seed)
			yield
	finally:
		random.setstate(python_state)
		np.random.set_state(numpy_state)
		if mps_state is not None:
			torch.mps.set_rng_state(mps_state)


@torch.no_grad()
def sample_keyed_batch(
		synthetic_model,
		batch_size,
		data_seed,
		step,
		*,
		autocast=False, ):
	"""Return deterministic observations ``x`` and true coefficients ``g``."""

	batch_size = int(batch_size)
	if batch_size < 1:
		raise ValueError("batch_size must be positive")
	device = synthetic_model.feature_dict.feature_vectors.device
	amp_enabled = bool(autocast and device.type in {"cpu", "cuda"})
	with (
		_isolated_rng(_batch_seed(data_seed, step), device),
		torch.autocast(
			device_type=device.type,
			dtype=torch.bfloat16,
			enabled=amp_enabled,
		),
	):
		g = synthetic_model.activation_generator.sample(batch_size)
		x = synthetic_model.feature_dict(g)
	if g.is_sparse:
		g = g.to_dense()
	x = x.to(torch.float32).reshape(batch_size, synthetic_model.feature_dict.feature_vectors.shape[1])
	g = g.to(torch.float32).reshape(batch_size, MODEL_NUM_FEATURES)
	return x, g


class SyntheticSource:
    """Independent reproducible training, prior, validation, and test streams."""

    def __init__(self, config, device="cpu", input_dim=768):
        from .normalization import ActivationNormalizer
        self.config = config
        self.generator = load_synthsaebench(
            config.get("benchmark", "historical"), snapshot=config.get("generator_path"),
            hidden_dim=input_dim, device=device)
        self.normalization_path = config.get("normalization_path")
        self.normalizer = ActivationNormalizer.from_artifact(
            config.get("normalization", "none"), self.normalization_path)
        self.device = torch.device(device)

    def sample_with_codes(self, batch_size, step, stream="train", seed=None):
        offsets = {"train": 0, "validation": 100003, "test": 200003, "prior": 300007}
        data_seed = int(self.config.get("seed", 1)) + offsets[stream] if seed is None else seed
        x, codes = sample_keyed_batch(self.generator, batch_size, data_seed, step)
        return self.normalizer.transform(x) * self.config.get("input_scale", 1.0), codes

    def sample(self, batch_size, step, stream="train"):
        return self.sample_with_codes(batch_size, step, stream)[0]
