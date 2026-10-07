"""Extract resumable, lossless BF16 Gemma activation shards in token order."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
import pathlib
import time

import torch

from befond.data import gemma_spec as spec
from befond.data.activation_cache import (
	cache_identity, file_sha256, load_activation_manifest, validate_shards,
)
from befond.data.gemma import GemmaActivationSource, load_data_manifest, load_gemma


def _save_manifest(path, manifest):
	temporary = path.with_suffix(".json.partial")
	with temporary.open("w") as file:
		json.dump(manifest, file, indent=2)
		file.write("\n")
		file.flush()
		os.fsync(file.fileno())
	os.replace(temporary, path)
	directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
	try:
		os.fsync(directory_fd)
	finally:
		os.close(directory_fd)


def _write_block(file, digest, cpu):
	buffer = memoryview(cpu.view(torch.uint16).numpy()).cast("B")
	file.write(buffer)
	digest.update(buffer)


def _write_shard(source, path, start, stop, block_tokens):
	"""Overlap the next Gemma forward with writing the preceding CPU block."""
	temporary = path.with_suffix(".bf16.partial")
	digest = hashlib.sha256()
	with temporary.open("wb") as file, ThreadPoolExecutor(max_workers=1) as writer:
		pending = None
		for offset in range(start, stop, block_tokens):
			x = source.read(offset, min(block_tokens, stop - offset))
			if x.dtype != torch.bfloat16 or not torch.isfinite(x).all().item():
				raise ValueError("Only finite native BF16 activations can be cached.")
			cpu = x.flatten(1).to("cpu").contiguous()
			if pending is not None:
				pending.result()
			pending = writer.submit(_write_block, file, digest, cpu)
		pending.result()
		file.flush()
		os.fsync(file.fileno())
	if temporary.stat().st_size != (stop - start) * spec.INPUT_DIM * 2:
		raise ValueError(f"Incomplete activation shard: {temporary}")
	checksum = digest.hexdigest()
	if file_sha256(temporary) != checksum:
		raise ValueError(f"Activation shard failed readback verification: {temporary}")
	os.replace(temporary, path)
	directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
	try:
		os.fsync(directory_fd)
	finally:
		os.close(directory_fd)
	return checksum


def extract_activations(data_dir, *, device="cuda:0", llm_batch_size=spec.TRAIN_LLM_BATCH_SIZE,
		shard_tokens=spec.ACTIVATION_SHARD_TOKENS, block_tokens=16_384,
		max_train_shards=None, verify_only=False, model=None):
	"""Resume at completed shard boundaries; never publish partially written shards."""
	if min(llm_batch_size, shard_tokens, block_tokens) < 1:
		raise ValueError("Extraction batch, shard, and block sizes must be positive.")
	if max_train_shards is not None and max_train_shards < 1:
		raise ValueError("max_train_shards must be positive.")
	data_dir = pathlib.Path(data_dir).expanduser().resolve()
	prepared = load_data_manifest(data_dir)
	cache_dir = data_dir / "activations"
	manifest_path = cache_dir / spec.ACTIVATION_MANIFEST
	if verify_only:
		manifest = load_activation_manifest(data_dir)
		for split in prepared["splits"]:
			count = validate_shards(cache_dir, manifest, split, verify=True)
			print(f"Verified {split}: {count:,}/{prepared['splits'][split]['num_tokens']:,} tokens", flush=True)
		return manifest
	cache_dir.mkdir(exist_ok=True)
	with (cache_dir / ".extract.lock").open("a") as lock:
		try:
			fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		except BlockingIOError:
			raise RuntimeError(f"Another extractor is already writing {cache_dir}.") from None
		runtime = {
			"llm_batch_size": llm_batch_size,
			"gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
			"torch": torch.__version__, "cuda": torch.version.cuda,
			"transformer_lens": version("transformer-lens"), "transformers": version("transformers"),
			"matmul_precision": torch.get_float32_matmul_precision(),
			"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
		}
		if manifest_path.exists():
			manifest = load_activation_manifest(data_dir)
			if manifest["extraction"] != runtime or manifest["shard_tokens"] != shard_tokens:
				raise ValueError("Resume requires the same extraction settings and package versions.")
		else:
			manifest = {
				"identity": cache_identity(prepared), "extraction": runtime,
				"shard_tokens": shard_tokens,
				"splits": {split: {"shards": [], "complete": False} for split in prepared["splits"]},
			}
			_save_manifest(manifest_path, manifest)
		# Validate completed prefixes before resuming; no Gemma load for a finished cache.
		completed = {split: validate_shards(cache_dir, manifest, split, verify=True)
			for split in prepared["splits"]}
		for split in ("validation", "calibration", "train"):
			info = manifest["splits"][split]
			total = prepared["splits"][split]["num_tokens"]
			limit = total if split != "train" or max_train_shards is None else min(
				total, max_train_shards * shard_tokens)
			if completed[split] >= limit:
				print(f"Already cached {split}: {completed[split]:,}/{total:,} tokens", flush=True)
				continue
			if model is None:
				model = load_gemma(device)
			source = GemmaActivationSource(data_dir, split=split, device=device,
				model=model, llm_batch_size=llm_batch_size, output_dtype=torch.bfloat16)
			(cache_dir / split).mkdir(exist_ok=True)
			for start in range(completed[split], limit, shard_tokens):
				stop = min(start + shard_tokens, total)
				path = cache_dir / split / f"{start:09d}-{stop:09d}.bf16"
				print(f"Extracting {split}: {start:,}–{stop:,}", flush=True)
				started = time.perf_counter()
				checksum = _write_shard(source, path, start, stop, block_tokens)
				info["shards"].append({"file": str(path.relative_to(cache_dir)),
					"start": start, "stop": stop, "sha256": checksum})
				info["complete"] = stop == total
				_save_manifest(manifest_path, manifest)
				elapsed = time.perf_counter() - started
				print(f"Cached {split}: {stop:,}/{total:,} tokens; "
					f"{(stop - start) / elapsed:,.0f} tokens/s including write and verification", flush=True)
			del source
		return manifest


def build_parser():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument("--data-dir", type=pathlib.Path, default=spec.DATA_DIR)
	parser.add_argument("--llm-batch-size", "--llm_batch_size", dest="llm_batch_size", type=int, default=spec.TRAIN_LLM_BATCH_SIZE)
	parser.add_argument("--max-train-shards", "--max_train_shards", dest="max_train_shards", type=int,
		help="Stop after this many total training shards; held-out splits are always included.")
	parser.add_argument("--verify-only", "--verify_only", dest="verify_only", action="store_true", help="Hash completed shards; do not load Gemma.")
	parser.add_argument("--show_config", "--dry_run", action="store_true")
	return parser


def main(argv=None):
	args = build_parser().parse_args(argv)
	if args.show_config:
		result = {**vars(args), "data_dir": str(args.data_dir),
			"shard_tokens": spec.ACTIVATION_SHARD_TOKENS, "storage_dtype": "bfloat16",
			"training_bytes": spec.PREPARED_TRAINING_TOKENS * spec.INPUT_DIM * 2}
		print(json.dumps(result, indent=2))
		return result
	if not args.verify_only and torch.device(args.device).type == "cuda":
		torch.cuda.set_device(args.device)
	torch.set_num_threads(2)
	torch.set_num_interop_threads(2)
	return extract_activations(args.data_dir, device=args.device,
		llm_batch_size=args.llm_batch_size, max_train_shards=args.max_train_shards,
		verify_only=args.verify_only)


if __name__ == "__main__":
	main()
