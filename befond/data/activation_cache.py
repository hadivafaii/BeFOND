"""Sequential BF16 activation shards with bounded CPU prefetch for BeFOND."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

from .gemma import load_data_manifest
from .gemma_spec import (
	ACTIVATION_MANIFEST, CONTEXT_LENGTH, EVAL_DEFAULTS, INPUT_DIM, model_provenance,
)


def file_sha256(path):
	with open(path, "rb") as file:
		return hashlib.file_digest(file, "sha256").hexdigest()


def cache_identity(data_manifest):
	return {
		"schema_version": 1, **model_provenance(),
		"prepared_data_manifest": data_manifest,
		"storage_dtype": "bfloat16", "byte_order": "little",
		"row_order": "split_active_token_order", "normalization": "none",
	}


def load_activation_manifest(data_dir):
	path = pathlib.Path(data_dir).expanduser().resolve() / "activations" / ACTIVATION_MANIFEST
	if not path.is_file():
		raise FileNotFoundError(
			f"Activation cache not found: {path}. Run "
			"python -m experiments.gemma.extract --device cuda:0 first.")
	manifest = json.loads(path.read_text())
	if manifest["identity"] != cache_identity(load_data_manifest(data_dir)):
		raise ValueError("Activation cache does not match the pinned model and prepared tokens.")
	if sys.byteorder != "little":
		raise ValueError("BF16 activation shards require a little-endian host.")
	if manifest["shard_tokens"] < 1:
		raise ValueError("Invalid activation shard size.")
	return manifest


def validate_shards(cache_dir, manifest, split, verify=False):
	"""Validate the completed prefix; hash full files only when explicitly requested."""
	info = manifest["splits"][split]
	total = manifest["identity"]["prepared_data_manifest"]["splits"][split]["num_tokens"]
	stop = 0
	for shard in info["shards"]:
		end = min(stop + manifest["shard_tokens"], total)
		filename = f"{split}/{stop:09d}-{end:09d}.bf16"
		if (shard["start"], shard["stop"], shard["file"]) != (stop, end, filename) or end <= stop:
			raise ValueError(f"Invalid or noncontiguous {split} activation shards.")
		path = cache_dir / filename
		if path.stat().st_size != (end - stop) * INPUT_DIM * 2:
			raise ValueError(f"Activation shard size mismatch: {path}")
		if verify and file_sha256(path) != shard["sha256"]:
			raise ValueError(f"Activation shard checksum mismatch: {path}")
		stop = end
	if info["complete"] != (stop == total):
		raise ValueError(f"Invalid {split} activation completion marker.")
	return stop


class CachedGemmaActivationSource:
	"""Map completed shards and copy sequential blocks through bounded pinned RAM."""

	def __init__(self, data_dir, split="train", device="cuda:0"):
		self.data_dir = pathlib.Path(data_dir).expanduser().resolve()
		self.cache_dir = self.data_dir / "activations"
		self.cache_manifest = load_activation_manifest(self.data_dir)
		self.manifest = self.cache_manifest["identity"]["prepared_data_manifest"]
		self.split, self.device = split, torch.device(device)
		self.split_manifest = self.manifest["splits"][split]
		self.num_tokens = validate_shards(self.cache_dir, self.cache_manifest, split)
		if not self.num_tokens:
			raise ValueError(f"No completed {split} activation shards; run extraction first.")
		self.input_dim, self.context_length = INPUT_DIM, CONTEXT_LENGTH
		self._shards = self.cache_manifest["splits"][split]["shards"]
		self._maps = {}
		self._executor = None
		self._pending = None

	def _check_span(self, start, count):
		if start < 0 or count < 1 or start + count > self.num_tokens:
			raise ValueError(f"Requested {self.split} span ({start}, {count}) exceeds "
				f"the {self.num_tokens:,} completed cached tokens.")

	def _read_cpu(self, start, count):
		cuda = self.device.type == "cuda"
		with torch.cuda.device(self.device) if cuda else nullcontext():
			output = torch.empty((count, INPUT_DIM), dtype=torch.bfloat16, pin_memory=cuda)
		written = 0
		while written < count:
			index = start // self.cache_manifest["shard_tokens"]
			shard = self._shards[index]
			if index not in self._maps:
				self._maps[index] = torch.from_file(str(self.cache_dir / shard["file"]),
					shared=False, size=(shard["stop"] - shard["start"]) * INPUT_DIM,
					dtype=torch.bfloat16).reshape(-1, INPUT_DIM)
			n = min(count - written, shard["stop"] - start)
			offset = start - shard["start"]
			np.copyto(output[written:written + n].view(torch.uint16).numpy(),
				self._maps[index][offset:offset + n].view(torch.uint16).numpy())
			start, written = start + n, written + n
		return output

	def read(self, start, count):
		start, count = int(start), int(count)
		self._check_span(start, count)
		if self._pending is not None:
			span, future = self._pending
			self._pending = None
			if span == (start, count):
				cpu = future.result()
			else:
				# Prior-feeder forks and seeks may interrupt the sequential stream.
				if not future.cancel():
					future.result()
				cpu = self._read_cpu(start, count)
		else:
			cpu = self._read_cpu(start, count)
		return cpu.to(self.device, dtype=torch.float32, non_blocking=True).reshape(
			count, INPUT_DIM)

	def prefetch(self, start, count):
		self._check_span(start, count)
		if self._pending is not None:
			if self._pending[0] == (start, count):
				return
			if not self._pending[1].cancel():
				self._pending[1].result()
		if self._executor is None:
			self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemma-cache")
		self._pending = ((start, count), self._executor.submit(self._read_cpu, start, count))

	def iter_activations(self, batch_size=EVAL_DEFAULTS["microbatch_size"], max_tokens=None):
		count = self.num_tokens if max_tokens is None else int(max_tokens)
		self._check_span(0, count)
		if batch_size < 1:
			raise ValueError("Activation batch size must be positive.")
		try:
			for start in range(0, count, batch_size):
				x = self.read(start, min(batch_size, count - start)).flatten(1)
				next_start = start + len(x)
				if next_start < count:
					self.prefetch(next_start, min(batch_size, count - next_start))
				yield x
		finally:
			self.close()

	def close(self):
		if self._executor is not None:
			self._executor.shutdown(wait=True, cancel_futures=True)
		self._executor, self._pending = None, None
		self._maps.clear()
