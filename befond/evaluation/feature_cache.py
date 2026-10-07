"""Task-local, lossless caches for frozen inference; never persist BeFOND codes."""

from contextlib import contextmanager
import hashlib
import pathlib
import tempfile
from unittest.mock import patch

import torch


def tensor_key(value):
	value = value.detach().cpu().contiguous()
	digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
	return str(value.dtype), tuple(value.shape), digest


class TensorStore:
	"""Bound resident code memory, spilling the remainder to temporary local disk."""

	def __init__(self, memory_bytes=256 * 1024 ** 2):
		self.memory_bytes = memory_bytes
		self.memory, self.paths = {}, {}
		self.resident_bytes = 0
		self.temporary = tempfile.TemporaryDirectory(prefix="befond-saebench-")
		self.serial = 0

	def __enter__(self):
		return self

	def __exit__(self, *_):
		self.memory.clear()
		self.paths.clear()
		self.temporary.cleanup()

	def __contains__(self, key):
		return key in self.memory or key in self.paths

	def put(self, key, value):
		# Copies are owned by the cache; upstream can mutate returned codes.
		value = value.detach().to(device="cpu", copy=True)
		size = value.numel() * value.element_size()
		if self.resident_bytes + size <= self.memory_bytes:
			self.memory[key] = value
			self.resident_bytes += size
		else:
			path = pathlib.Path(self.temporary.name) / f"{self.serial}.pt"
			self.serial += 1
			torch.save(value, path)
			self.paths[key] = path

	@torch.inference_mode(False)
	@torch.no_grad()
	def get(self, key, device="cpu"):
		if key in self.memory:
			return self.memory[key].to(device=device, copy=True)
		return torch.load(self.paths[key], map_location=device, weights_only=True)

	def discard(self, key):
		if key in self.memory:
			value = self.memory.pop(key)
			self.resident_bytes -= value.numel() * value.element_size()
		if key in self.paths:
			self.paths.pop(key).unlink()


@contextmanager
def cached_encode(adapter):
	"""Reuse only identical whole batches, without changing inference arithmetic."""
	encode = adapter.encode
	with TensorStore() as cache:
		def memoized(x):
			key = tensor_key(x)
			if key in cache:
				return cache.get(key, adapter.device)
			codes = encode(x)
			cache.put(key, codes)
			return codes
		with patch.object(adapter, "encode", memoized):
			yield
