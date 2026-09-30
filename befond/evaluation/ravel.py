"""Official RAVEL mask training with pinned HF Gemma and frozen BeFOND inference."""

from dataclasses import asdict, replace
from contextlib import contextmanager
import pathlib
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from .protocol import _output_fields, _seed
from .protocol import task_seed
from befond.data.gemma_spec import HF_MODEL_NAME, LLM_DTYPE, MODEL_REVISION


RAVEL_DATASET_REVISION = "dead73bd9a75a3d5a54bd565c4391eb936830248"


@contextmanager
def cached_ravel_features(adapter):
	"""Keep original batch layouts, mask training, and decoder gradients intact."""
	from sae_bench.evals.ravel import main as upstream
	from sae_bench.sae_bench_utils import activation_collection
	from .feature_cache import TensorStore, cached_encode, tensor_key

	run_attribute = upstream.run_eval_single_cause_attribute
	get_activations = activation_collection.get_layer_activations
	def run(*args, **kwargs):
		# Discard features between attributes rather than retaining an entire entity.
		with TensorStore() as sources, cached_encode(adapter):
			def source_activations(model, layer, encoding, positions):
				key = (layer, tuple((name, tensor_key(value)) for name, value in sorted(encoding.items())),
					tensor_key(positions))
				if key in sources:
					return sources.get(key, model.device)
				value = get_activations(model, layer, encoding, positions)
				sources.put(key, value)
				return value
			with patch.object(activation_collection, "get_layer_activations", source_activations):
				return run_attribute(*args, **kwargs)
	with patch.object(upstream, "run_eval_single_cause_attribute", run):
		yield


def ravel_config(args):
	from sae_bench.evals.ravel.eval_config import RAVELEvalConfig

	return RAVELEvalConfig(model_name=HF_MODEL_NAME, random_seed=args.seed,
		llm_dtype=LLM_DTYPE, llm_batch_size=args.ravel_batch_size)


def ravel_tasks(config):
	"""Keep all cause/isolation attributes together and seed each entity independently."""
	return {entity: replace(config, entity_attribute_selection={entity: list(attributes)},
		random_seed=task_seed(config.random_seed, f"ravel/{entity}"))
		for entity, attributes in config.entity_attribute_selection.items()}


def merge_ravel(config, payloads):
	"""Preserve upstream's equal-entity mean of attribute-averaged scores."""
	tasks = ravel_tasks(config)
	if not tasks or set(payloads) != set(tasks):
		raise ValueError("RAVEL results must contain every configured entity exactly once.")
	for entity, task in tasks.items():
		if payloads[entity]["eval_config"]["entity_attribute_selection"] != task.entity_attribute_selection:
			raise ValueError(f"RAVEL result does not match entity task {entity!r}.")
	ordered = [payloads[entity] for entity in tasks]
	metrics = {key: sum(payload["eval_result_metrics"]["ravel"][key] for payload in ordered) / len(ordered)
		for key in ("disentanglement_score", "cause_score", "isolation_score")}
	unstructured = {f"{entity}_results": payloads[entity]["eval_result_unstructured"][f"{entity}_results"]
		for entity in tasks}
	unstructured["execution"] = {
		**ordered[0]["eval_result_unstructured"]["execution"],
		"task_seeds": {entity: task.random_seed for entity, task in tasks.items()},
		"task_execution": {entity: payloads[entity]["eval_result_unstructured"]["execution"] for entity in tasks},
	}
	return {**ordered[0], "eval_config": asdict(config),
		"eval_result_metrics": {"ravel": metrics},
		"eval_result_details": [detail for payload in ordered for detail in payload["eval_result_details"]],
		"eval_result_unstructured": unstructured}


def load_ravel_model(device):
	"""RAVEL uses Hugging Face residual hooks, with gradients only for its mask."""
	from transformers import AutoModelForCausalLM

	model = AutoModelForCausalLM.from_pretrained(
		HF_MODEL_NAME, revision=MODEL_REVISION, device_map=str(device),
		torch_dtype=getattr(torch, LLM_DTYPE), attn_implementation="eager")
	return model.eval().requires_grad_(False)


class _ResidualSAE(nn.Module):
	"""Cast reconstruction at the HF hook boundary without lowering BeFOND precision."""

	def __init__(self, adapter, dtype):
		super().__init__()
		self.adapter = adapter
		self.cfg = adapter.cfg
		self.residual_dtype = dtype

	def encode(self, inputs):
		return self.adapter.encode(inputs)

	def decode(self, codes):
		return self.adapter.decode(codes).to(dtype=self.residual_dtype)

	def train(self, mode=True):
		# MDBM.train() recursively visits its SAE; inference must stay deterministic.
		return super().train(False)


class _TupleHookLayer:
	"""Expose the tuple convention expected by upstream across Transformers versions."""

	def __init__(self, layer):
		self.layer = layer
		self.handles = []

	def register_forward_hook(self, hook):
		def apply(module, inputs, outputs):
			is_tensor = isinstance(outputs, torch.Tensor)
			result = hook(module, inputs, (outputs,) if is_tensor else outputs)
			return result[0] if is_tensor and result is not None else result
		handle = self.layer.register_forward_hook(apply)
		self.handles.append(handle)
		return handle


class _RAVELModel(nn.Module):
	"""Adapt only RAVEL's layer hooks; the underlying HF forward stays unchanged."""

	def __init__(self, model):
		super().__init__()
		self.raw_model = model
		self.model = SimpleNamespace(layers=[_TupleHookLayer(layer) for layer in model.model.layers])
		self.config = model.config

	@property
	def device(self):
		return self.raw_model.device

	@property
	def dtype(self):
		return self.raw_model.dtype

	def forward(self, *args, **kwargs):
		return self.raw_model(*args, **kwargs)

	def generate(self, *args, **kwargs):
		return self.raw_model.generate(*args, **kwargs)

	def train(self, mode=True):
		return super().train(False)

	def remove_hooks(self):
		# Upstream removes hooks after successful forwards, but not after exceptions.
		for layer in self.model.layers:
			for handle in layer.handles:
				handle.remove()
			layer.handles.clear()


def _prepare_datasets(config, model=None, get_llm=None):
	"""Build all model-only caches before seeding mask training, including on resume."""
	from sae_bench.evals.ravel.instance import RAVELInstance, get_instance_name
	from transformers import AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
	for entity in config.entity_attribute_selection:
		path = pathlib.Path(config.artifact_dir) / get_instance_name(
			entity, config.model_name, config.full_dataset_downsample, config.top_n_entities)
		if path.exists():
			continue
		if model is None:
			model = _RAVELModel(get_llm("ravel"))
		_seed(config.random_seed)
		with torch.no_grad():
			instance = RAVELInstance.create_from_files(
				config=replace(config, llm_batch_size=config.llm_batch_size * 8),
				entity_type=entity, data_dir=config.artifact_dir, tokenizer=tokenizer,
				model=model, model_name=config.model_name,
				attribute_types=config.entity_attribute_selection[entity],
				downsample=config.full_dataset_downsample)
			instance.create_and_save_filtered_dataset(
				artifact_dir=config.artifact_dir, top_n_entities=config.top_n_entities)


def prepare_ravel_cache(config, get_llm, cache, model=None):
	"""Prepare each shared entity once before scheduling masks across variants."""
	from huggingface_hub import snapshot_download

	cache = pathlib.Path(cache) / RAVEL_DATASET_REVISION
	cache.mkdir(parents=True, exist_ok=True)
	# Upstream repeatedly loads AutoTokenizer(config.model_name) without revision.
	# A pinned local snapshot makes every one of those loads reproducible.
	tokenizer_path = snapshot_download(HF_MODEL_NAME, revision=MODEL_REVISION,
		allow_patterns=["tokenizer*", "*.model", "special_tokens_map.json", "config.json"])
	snapshot_download("adamkarvonen/ravel_prompts", repo_type="dataset",
		revision=RAVEL_DATASET_REVISION, local_dir=str(cache / "base"), allow_patterns="*.json")
	runtime_config = replace(config, model_name=tokenizer_path, artifact_dir=str(cache))
	_prepare_datasets(runtime_config, model, get_llm)
	return runtime_config


def run_ravel(config, adapter, model, cache, variant_id):
	"""Use upstream pair sampling, mask optimization, generation, and scoring."""
	from sae_bench.evals.ravel.main import run_eval_single_sae
	from sae_bench.evals.ravel.eval_output import (
		RAVELEvalOutput, RAVELMetricCategories, RAVELMetricResults,
	)

	wrapped_model = _RAVELModel(model)
	try:
		runtime_config = prepare_ravel_cache(config, None, cache, wrapped_model)
		_seed(config.random_seed)
		# Upstream enables global gradient mode. Contain it, while allowing decoder
		# gradients into the trained mask and keeping LLM/BeFOND weights frozen.
		with torch.enable_grad(), cached_ravel_features(adapter):
			results, per_class = run_eval_single_sae(runtime_config,
				_ResidualSAE(adapter, model.dtype), wrapped_model, str(adapter.device), runtime_config.artifact_dir)
	finally:
		wrapped_model.remove_hooks()
		model.eval()
		adapter.eval()
	return asdict(RAVELEvalOutput(
		**_output_fields(config, adapter, variant_id),
		eval_result_metrics=RAVELMetricCategories(ravel=RAVELMetricResults(
			disentanglement_score=results["disentangle_score"],
			cause_score=results["cause_score"], isolation_score=results["isolation_score"])),
		eval_result_details=[],
		eval_result_unstructured={**per_class, "execution": {
			"model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
			"dataset_revision": RAVEL_DATASET_REVISION, "artifact_dir": runtime_config.artifact_dir,
			"model_loader": "AutoModelForCausalLM.from_pretrained",
			"sae_dtype": str(adapter.dtype), "residual_dtype": str(model.dtype),
		}}))
