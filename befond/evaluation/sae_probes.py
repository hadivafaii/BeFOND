"""Current SAEBench SAE Probes protocol with pinned model activations."""

from dataclasses import asdict, replace
from contextlib import ExitStack, contextmanager
import fcntl
import importlib
import json
import pathlib
import shutil
from unittest.mock import patch

from befond.data.gemma_spec import HOOK_NAME, MODEL_NAME


def sae_probes_config(args):
	from sae_bench.evals.sparse_probing_sae_probes.eval_config import SparseProbingSaeProbesEvalConfig

	config = SparseProbingSaeProbesEvalConfig(model_name=MODEL_NAME, random_seed=args.seed)
	if args.sae_probes_datasets is not None:
		unknown = set(args.sae_probes_datasets) - set(config.dataset_names)
		if unknown:
			raise ValueError(f"Unknown SAE Probes datasets: {sorted(unknown)}")
		config.dataset_names = [name for name in config.dataset_names if name in args.sae_probes_datasets]
	return config


def sae_probes_tasks(config):
	"""Keep the protocol seed while scheduling each dataset independently."""
	if not config.dataset_names or len(set(config.dataset_names)) != len(config.dataset_names):
		raise ValueError("SAE Probes requires a nonempty list of distinct datasets.")
	return {name: replace(config, dataset_names=[name]) for name in config.dataset_names}


def merge_sae_probes(config, payloads):
	"""Reproduce upstream's per-metric dataset means from singleton results."""
	from sae_bench.evals.sparse_probing_sae_probes.eval_output import (
		SaeProbesLlmMetrics, SaeProbesSaeMetrics,
	)

	expected = sae_probes_tasks(config)
	if set(payloads) != set(expected):
		raise RuntimeError("SAE Probes returned an incomplete or unexpected set of tasks.")
	details = []
	# Upstream reads sorted JSON paths; retain its detail and summation order.
	for name in sorted(expected):
		task_details = payloads[name]["eval_result_details"]
		if len(task_details) != 1 or task_details[0]["dataset_name"] != name:
			raise RuntimeError(f"SAE Probes returned invalid task details for {name!r}.")
		details.extend(task_details)

	def mean(values):
		values = [value for value in values if value is not None]
		return float(sum(values) / len(values)) if values else None

	aggregate, by_k = {}, {}
	for k in config.ks:
		metrics = []
		for detail in details:
			per_k = detail.get("sae_metrics_by_k") or {}
			metrics.append(per_k.get(k, per_k.get(str(k), {})))
		k_metrics = {}
		for metric in ("test_accuracy", "test_auc", "test_f1"):
			value = mean(task.get(metric) for task in metrics)
			aggregate[f"sae_top_{k}_{metric}"] = value
			if value is not None:
				k_metrics[metric] = value
		if k_metrics:
			by_k[k] = k_metrics
	llm = {}
	if config.include_llm_baseline:
		for metric in ("llm_test_accuracy", "llm_test_auc", "llm_test_f1"):
			llm[metric] = mean(detail.get(metric) for detail in details)
	return {
		**next(iter(payloads.values())),
		"eval_config": asdict(config),
		"eval_result_details": details,
		"eval_result_metrics": {
			"llm": asdict(SaeProbesLlmMetrics(**llm)),
			"sae": asdict(SaeProbesSaeMetrics(**aggregate)),
		},
		"sae_metrics_by_k": by_k or None,
	}


def prepare_sae_probes_cache(config, get_llm, device, cache):
	"""Generate model activations once before concurrent BeFOND variants read them."""
	from sae_probes.generate_model_activations import ensure_dataset_activations

	model_cache = pathlib.Path(cache) / "model_activations"
	missing = [dataset for dataset in config.dataset_names
		if not (model_cache / f"model_activations_{MODEL_NAME}" / f"{dataset}_{HOOK_NAME}.pt").exists()]
	if missing:
		llm = get_llm()
		tokenizer = llm.tokenizer
		padding, truncation = tokenizer.padding_side, tokenizer.truncation_side
		try:
			ensure_dataset_activations(
				model_name=MODEL_NAME, dataset_short_names=missing, hook_names=[HOOK_NAME],
				model_cache_path=model_cache, device=str(device), model=llm)
		finally:
			tokenizer.padding_side, tokenizer.truncation_side = padding, truncation
	return model_cache


def _shared_baselines(config, cache, raw, run_baselines, **kwargs):
	"""Reuse model-only fits, exposing only this task's JSONs to upstream."""
	from .protocol import identity, _save_json
	import torch
	from threadpoolctl import threadpool_info

	settings = {name: getattr(config, name) for name in (
		"model_name", "random_seed", "setting", "baseline_method")}
	settings["torch_threads"] = torch.get_num_threads()
	settings["threadpools"] = [{field: pool.get(field) for field in (
		"internal_api", "prefix", "version", "num_threads", "architecture")}
		for pool in threadpool_info()]
	key = identity(settings)
	shared = pathlib.Path(cache) / "baselines" / key
	shared.mkdir(parents=True, exist_ok=True)
	for dataset in config.dataset_names:
		# One lock per dataset also protects simultaneous evaluation processes.
		with (shared / f"{identity(dataset)}.lock").open("a") as lock:
			fcntl.flock(lock, fcntl.LOCK_EX)
			run_baselines(**{**kwargs, "datasets": [dataset], "results_path": str(shared)})
			# Retain upstream's directory/file names, but never aggregate sibling tasks.
			for source in shared.glob("baseline_results_*/**/*.json"):
				if source.name.startswith(f"{dataset}_{HOOK_NAME}_"):
					target = pathlib.Path(raw) / source.relative_to(shared)
					target.parent.mkdir(parents=True, exist_ok=True)
					_save_json(target, json.loads(source.read_text()))


@contextmanager
def activation_loading(config, split="test"):
	"""Apply the same input promotion and validation split in both execution paths."""
	import torch
	from sae_probes import utils_data

	get_xvals = utils_data.get_xvals
	def load_activations(*args, **kwargs):
		activations = get_xvals(*args, **kwargs)
		return activations.float() if activations.dtype == torch.bfloat16 else activations
	with ExitStack() as context:
		context.enter_context(patch.object(utils_data, "get_xvals", load_activations))
		if split == "validation":
			if config.setting != "normal" or config.include_llm_baseline:
				raise ValueError("Validation requires normal probes without the LLM baseline.")
			from sklearn.model_selection import train_test_split
			activations = importlib.import_module("sae_probes.generate_sae_activations")
			get_train_test = activations.get_xy_traintest
			def validation_data(*args, **kwargs):
				x, y, _, _ = get_train_test(*args, **kwargs)
				fit, validation = train_test_split(list(range(len(y))), test_size=0.1,
					stratify=y, random_state=config.random_seed)
				return x[fit], y[fit], x[validation], y[validation]
			context.enter_context(patch.object(activations, "get_xy_traintest", validation_data))
		yield


def run_sae_probes(config, adapter, get_llm, cache, output, variant_id, force=False, *, split="test"):
	"""Delegate selection, validation, scoring and aggregation to upstream.

	Raw model activations are shared; result JSONs belong to this full protocol
	and BeFOND variant. The upstream JSON names alone do not encode seeds or K.
	"""
	import torch
	from sae_bench.evals.sparse_probing_sae_probes import main as upstream
	if split not in ("test", "validation"):
		raise ValueError(f"Unknown SAE Probes split: {split!r}")

	output = pathlib.Path(output)
	model_cache = prepare_sae_probes_cache(config, get_llm, adapter.device, cache)
	# Upstream's force_rerun does not invalidate its underlying per-task JSONs.
	# This directory contains only derived results for this exact variant.
	raw = output / "raw"
	if force and raw.exists():
		shutil.rmtree(raw)
	execution_config = replace(config, results_path=str(raw), model_cache_path=str(model_cache))
	# Workers run one task at a time; restore the upstream loader even on failure.
	with ExitStack() as split_context, activation_loading(config, split), \
			torch.inference_mode(False), torch.enable_grad():
		if config.include_llm_baseline:
			run_baselines = upstream.run_baseline_evals
			split_context.enter_context(patch.object(upstream, "run_baseline_evals",
				lambda **kwargs: _shared_baselines(config, cache, raw, run_baselines, **kwargs)))
		results = upstream.run_eval(execution_config, [(variant_id, adapter)], str(adapter.device),
			str(output / "upstream"), force_rerun=True)
	payload, = results.values()
	details = payload["eval_result_details"]
	if {detail["dataset_name"] for detail in details} != set(config.dataset_names):
		raise RuntimeError("SAE Probes returned an incomplete set of task results.")
	return payload
