"""Prepare compact token mmaps for Gemma activation extraction."""

import argparse
import hashlib
import json
import pathlib
import tempfile

import numpy as np

from befond.data import gemma_spec as spec
from befond.data.gemma_spec import (
	CONTEXT_LENGTH, DATA_MANIFEST, DATASET_NAME, DATASET_REVISION,
	HELDOUT_TOKENS, HF_MODEL_NAME, MODEL_REVISION,
	PREPARED_TRAINING_TOKENS,
)


def prepare_split(documents, tokenizer, directory, split, num_tokens):
	"""Consume whole documents per split; discard truncated tails at boundaries."""
	tokens_path = directory / f"{split}.tokens.uint32"
	lengths_path = directory / f"{split}.lengths.uint32"
	token_hash, length_hash = hashlib.sha256(), hashlib.sha256()
	active_tokens, document_count, used_documents = 0, 0, 0
	with open(tokens_path, "wb") as tokens_file, open(lengths_path, "wb") as lengths_file:
		while active_tokens < num_tokens:
			document = next(documents)
			document_count += 1
			ids = tokenizer(
				document["text"], add_special_tokens=True,
				truncation=True, max_length=CONTEXT_LENGTH,
			)["input_ids"]
			if not ids or ids[0] != tokenizer.bos_token_id:
				raise ValueError("Gemma tokenization must begin with its BOS token.")
			if len(ids) < 2:
				continue
			# The last document contributes only the requested active-token budget.
			ids = ids[:min(len(ids), num_tokens - active_tokens + 1)]
			token_bytes = np.asarray(ids, dtype="<u4").tobytes()
			length_bytes = np.asarray([len(ids)], dtype="<u4").tobytes()
			tokens_file.write(token_bytes)
			lengths_file.write(length_bytes)
			token_hash.update(token_bytes)
			length_hash.update(length_bytes)
			active_tokens += len(ids) - 1
			used_documents += 1
	return {
		"tokens_file": tokens_path.name,
		"lengths_file": lengths_path.name,
		"tokens_sha256": token_hash.hexdigest(),
		"lengths_sha256": length_hash.hexdigest(),
		"num_tokens": active_tokens,
		"num_contexts": used_documents,
		"source_documents_consumed": document_count,
	}


def prepare_data(data_dir, *, training_tokens=PREPARED_TRAINING_TOKENS,
		validation_tokens=HELDOUT_TOKENS, calibration_tokens=HELDOUT_TOKENS):
	"""Stream the pinned Pile once, reserving fixed held-out documents first."""
	from datasets import load_dataset
	from transformers import AutoTokenizer

	data_dir = pathlib.Path(data_dir).expanduser().resolve()
	if data_dir.exists():
		raise FileExistsError(f"Refusing to replace prepared data: {data_dir}")
	budgets = {"validation": validation_tokens, "calibration": calibration_tokens,
		"train": training_tokens}
	if any(int(value) < 1 for value in budgets.values()):
		raise ValueError("All three token budgets must be positive.")
	tokenizer = AutoTokenizer.from_pretrained(
		HF_MODEL_NAME, revision=MODEL_REVISION)
	documents = iter(load_dataset(
		DATASET_NAME, revision=DATASET_REVISION, split="train", streaming=True))
	manifest = {
		"schema_version": 1,
		"dataset_name": DATASET_NAME,
		"dataset_revision": DATASET_REVISION,
		"dataset_split": "train",
		"document_order": "source_order",
		"tokenizer_name": HF_MODEL_NAME,
		"tokenizer_revision": MODEL_REVISION,
		"context_length": CONTEXT_LENGTH,
		"token_dtype": "<u4",
		"bos_token_id": tokenizer.bos_token_id,
		"pad_token_id": tokenizer.pad_token_id,
		"tokenization": "per_document_truncate_with_bos_no_packing",
		"active_tokens": "all_document_tokens_except_initial_bos",
		"split_order": list(budgets),
		"splits": {},
	}
	data_dir.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix=".saebench-", dir=data_dir.parent) as temporary:
		staging = pathlib.Path(temporary) / "data"
		staging.mkdir()
		document_start = 0
		for split, budget in budgets.items():
			info = prepare_split(documents, tokenizer, staging, split, int(budget))
			info["source_document_start"] = document_start
			document_start += info["source_documents_consumed"]
			info["source_document_stop"] = document_start
			manifest["splits"][split] = info
		(staging / DATA_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
		staging.rename(data_dir)
	return manifest


def build_parser():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/gemma.json")
	parser.add_argument("--data-dir", type=pathlib.Path)
	parser.add_argument("--training-tokens", type=int)
	parser.add_argument("--validation-tokens", "--validation_tokens", dest="validation_tokens", type=int, default=HELDOUT_TOKENS)
	parser.add_argument("--calibration-tokens", "--calibration_tokens", dest="calibration_tokens", type=int, default=HELDOUT_TOKENS)
	parser.add_argument("--show_config", "--dry_run", action="store_true")
	return parser


def main(argv=None):
	from befond.config import config_path
	args = build_parser().parse_args(argv)
	data = json.loads(config_path(args.config).read_text())["data"]
	args.data_dir = args.data_dir or pathlib.Path(data["path"])
	args.training_tokens = args.training_tokens or data.get("prepared_training_tokens", PREPARED_TRAINING_TOKENS)
	if args.show_config:
		print(json.dumps(vars(args), default=str, indent=2))
		return vars(args)
	manifest = prepare_data(args.data_dir, training_tokens=args.training_tokens,
		validation_tokens=args.validation_tokens, calibration_tokens=args.calibration_tokens)
	print(f"Prepared tokens: {args.data_dir.expanduser().resolve()}")
	return manifest


if __name__ == "__main__":
	main()
