"""Pinned Gemma activations over document-disjoint, bounded token streams."""

import hashlib
import json
import pathlib

import numpy as np
import torch

from .gemma_spec import (
	ACTIVATION_BUFFER_TOKENS, CONTEXT_LENGTH, DATA_MANIFEST, DATA_SEED,
	DATASET_NAME, DATASET_REVISION, EVAL_DEFAULTS,
	HF_MODEL_NAME, HOOK_LAYER, HOOK_NAME, INPUT_DIM, LLM_DTYPE, MODEL_NAME,
	MODEL_REVISION, TRAIN_LLM_BATCH_SIZE,
)


def load_gemma(device, dtype=LLM_DTYPE):
	"""Use the same activation coordinates as upstream sparse probing."""
	from transformer_lens import HookedTransformer

	if isinstance(dtype, str):
		dtype = getattr(torch, dtype)
	model = HookedTransformer.from_pretrained_no_processing(
		MODEL_NAME, revision=MODEL_REVISION, device=str(device), dtype=dtype)
	return model.eval().requires_grad_(False)


def load_data_manifest(data_dir):
	data_dir = pathlib.Path(data_dir).expanduser().resolve()
	manifest = json.loads((data_dir / DATA_MANIFEST).read_text())
	expected = {
		"schema_version": 1,
		"dataset_name": DATASET_NAME,
		"dataset_revision": DATASET_REVISION,
		"tokenizer_name": HF_MODEL_NAME,
		"tokenizer_revision": MODEL_REVISION,
		"context_length": CONTEXT_LENGTH,
		"token_dtype": "<u4",
		"tokenization": "per_document_truncate_with_bos_no_packing",
		"active_tokens": "all_document_tokens_except_initial_bos",
	}
	for key, value in expected.items():
		if manifest.get(key) != value:
			raise ValueError(f"Prepared data {key} does not match this Gemma protocol.")
	previous_stop = 0
	for split in ("validation", "calibration", "train"):
		info = manifest["splits"][split]
		start, stop = info["source_document_start"], info["source_document_stop"]
		if start != previous_stop or stop <= start:
			raise ValueError("Prepared document splits must be ordered and disjoint.")
		previous_stop = stop
	return manifest


class GemmaActivationSource:
	"""Extract at most a few contexts at a time; never materialize the full pilot."""

	def __init__(self, data_dir, split="train", device="cuda:0", model=None,
			llm_batch_size=TRAIN_LLM_BATCH_SIZE, output_dtype=torch.float32):
		self.data_dir = pathlib.Path(data_dir).expanduser().resolve()
		self.manifest = load_data_manifest(self.data_dir)
		self.split = split
		self.split_manifest = self.manifest["splits"][split]
		self.device = torch.device(device)
		self.model = model
		self.output_dtype = output_dtype
		self.llm_batch_size = int(llm_batch_size)
		if self.llm_batch_size < 1:
			raise ValueError("llm_batch_size must be positive.")
		self.num_tokens = int(self.split_manifest["num_tokens"])
		self.context_length = CONTEXT_LENGTH
		self.input_dim = INPUT_DIM
		paths = {}
		for name in ("tokens", "lengths"):
			path = self.data_dir / self.split_manifest[f"{name}_file"]
			with open(path, "rb") as file:
				digest = hashlib.file_digest(file, "sha256").hexdigest()
			if digest != self.split_manifest[f"{name}_sha256"]:
				raise ValueError(f"Prepared {split} {name} checksum mismatch.")
			paths[name] = path
		self.lengths = np.memmap(paths["lengths"], mode="r", dtype="<u4")
		if len(self.lengths) != self.split_manifest["num_contexts"] or not len(self.lengths):
			raise ValueError("Prepared document count does not match its manifest.")
		if np.any(self.lengths < 2) or np.any(self.lengths > CONTEXT_LENGTH):
			raise ValueError("Invalid prepared context length.")
		self.token_offsets = np.concatenate(([0], np.cumsum(self.lengths, dtype=np.int64)))
		self.active_offsets = self.token_offsets - np.arange(len(self.token_offsets))
		self.tokens = np.memmap(paths["tokens"], mode="r", dtype="<u4")
		if len(self.tokens) != self.token_offsets[-1] or self.active_offsets[-1] != self.num_tokens:
			raise ValueError("Prepared token count does not match its manifest.")
		if np.any(self.tokens[self.token_offsets[:-1]] != self.manifest["bos_token_id"]):
			raise ValueError("Prepared documents must begin with BOS.")
		self._cached_batch_start = None
		self._cached_contexts = None

	def _context_activations(self, context_index):
		batch_start = context_index // self.llm_batch_size * self.llm_batch_size
		if batch_start != self._cached_batch_start:
			if self.model is None:
				self.model = load_gemma(self.device)
			batch_stop = min(batch_start + self.llm_batch_size, len(self.lengths))
			lengths = self.lengths[batch_start:batch_stop].astype(np.int64)
			ids = np.full((len(lengths), int(lengths.max())),
				self.manifest["pad_token_id"], dtype=np.int64)
			for row, index in enumerate(range(batch_start, batch_stop)):
				ids[row, :lengths[row]] = self.tokens[
					self.token_offsets[index]:self.token_offsets[index + 1]]
			attention_mask = torch.from_numpy(
				(np.arange(ids.shape[1])[None] < lengths[:, None]).astype(np.int64)).to(self.device)
			with torch.inference_mode():
				_, cache = self.model.run_with_cache(
					torch.from_numpy(ids).to(self.device),
					attention_mask=attention_mask, prepend_bos=False,
					names_filter=[HOOK_NAME], stop_at_layer=HOOK_LAYER + 1,
					return_type=None)
				captured = cache[HOOK_NAME]
			if captured.shape != (*ids.shape, INPUT_DIM):
				raise ValueError(f"Unexpected Gemma hook shape: {tuple(captured.shape)}")
			if self.output_dtype == torch.bfloat16 and captured.dtype != torch.bfloat16:
				raise ValueError("BF16 caching requires native BF16 hook outputs; refusing lossy conversion.")
			# Clone outside inference_mode so callers receive ordinary tensors.
			self._cached_contexts = [
				captured[row, 1:length].detach().to(dtype=self.output_dtype, copy=True)
				for row, length in enumerate(lengths)]
			self._cached_batch_start = batch_start
		return self._cached_contexts[context_index - batch_start]

	def read(self, start, count):
		start, count = int(start), int(count)
		if start < 0 or count < 1 or start + count > self.num_tokens:
			raise ValueError(f"Invalid {self.split} activation span: {start}, {count}.")
		stop, pieces = start + count, []
		while start < stop:
			index = int(np.searchsorted(self.active_offsets, start, side="right") - 1)
			offset = start - int(self.active_offsets[index])
			n = min(stop - start, int(self.lengths[index]) - 1 - offset)
			pieces.append(self._context_activations(index)[offset:offset + n])
			start += n
		x = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
		return x.reshape(count, INPUT_DIM)

	def iter_activations(self, batch_size=EVAL_DEFAULTS["microbatch_size"], max_tokens=None):
		"""Deterministic raw vectors, suitable for repeatable threshold calibration."""
		count = self.num_tokens if max_tokens is None else int(max_tokens)
		if not 0 < count <= self.num_tokens or batch_size < 1:
			raise ValueError("Invalid activation iterator token or batch count.")
		for start in range(0, count, batch_size):
			yield self.read(start, min(batch_size, count - start)).flatten(1)


class CachedActivationStream:
    """Read a finite token prefix with deterministic, bounded block shuffling."""

    def __init__(self, config, device="cpu"):
        from .activation_cache import CachedGemmaActivationSource
        from .normalization import ActivationNormalizer
        self.config = config
        self.device = torch.device(device)
        self.source = CachedGemmaActivationSource(config["path"], device=device)
        self.num_tokens = int(config.get("training_tokens", self.source.num_tokens))
        if not 0 < self.num_tokens <= self.source.num_tokens:
            raise ValueError("Requested training prefix exceeds prepared activations.")
        self.normalization_path = config.get("normalization_path")
        self.normalizer = ActivationNormalizer.from_artifact(
            config.get("normalization", "none"), self.normalization_path)
        self.buffer_tokens = int(config.get("buffer_tokens", 16384))
        self.seed = int(config.get("seed", 0))
        self._blocks = {}
        self._validation = None

    def sample(self, batch_size, step, stream="train"):
        if stream in ("validation", "test"):
            if self._validation is None:
                from .activation_cache import CachedGemmaActivationSource
                self._validation = CachedGemmaActivationSource(
                    self.config["path"], split="validation", device=self.device)
            start = (step * batch_size) % self._validation.num_tokens
            count = min(batch_size, self._validation.num_tokens - start)
            return self.normalizer.transform(self._validation.read(start, count))
        # The separate prior feeder revisits the same train prefix, on its own step counter.
        batches = self.num_tokens // batch_size
        if batches < 1:
            raise ValueError("Training prefix is smaller than one batch.")
        start = (int(step) % batches) * batch_size
        if self.buffer_tokens == 0:
            return self.normalizer.transform(self.source.read(start, batch_size))
        stop, pieces = start + batch_size, []
        while start < stop:
            block, offset = divmod(start, self.buffer_tokens)
            saved = self._blocks.get(stream)
            if saved is None or saved[0] != block:
                block_start = block * self.buffer_tokens
                count = min(self.buffer_tokens, self.num_tokens - block_start)
                values = self.source.read(block_start, count)
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, block]))
                order = torch.from_numpy(rng.permutation(count)).to(values.device)
                saved = (block, values[order])
                self._blocks[stream] = saved
            count = min(stop - start, len(saved[1]) - offset)
            pieces.append(saved[1][offset:offset + count])
            start += count
        return self.normalizer.transform(torch.cat(pieces))
