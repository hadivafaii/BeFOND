"""Public model/data revisions and token extraction protocol."""

import pathlib

DATA_DIR = pathlib.Path("data/gemma")

EVAL_DEFAULTS = {"microbatch_size": 256}

DATASET_KEY = "Gemma2-2B-Resid12"

TRAINING_TOKENS = 50_000_000

NUM_LATENTS = 16_384

PREPARED_TRAINING_TOKENS = 500_000_000

ACTIVATION_SHARD_TOKENS = 1_048_576

HELDOUT_TOKENS = 65_536

CONTEXT_LENGTH = 1024

DATA_SEED = 0

TRAIN_LLM_BATCH_SIZE = 4

ACTIVATION_BUFFER_TOKENS = 16_384

MODEL_NAME = "gemma-2-2b"

HF_MODEL_NAME = "google/gemma-2-2b"

MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"

HOOK_LAYER = 12

HOOK_NAME = f"blocks.{HOOK_LAYER}.hook_resid_post"

INPUT_DIM = 2304

DATASET_NAME = "monology/pile-uncopyrighted"

DATASET_REVISION = "3be90335b66f24456a5d6659d9c8d208c0357119"

LLM_DTYPE = "bfloat16"

LLM_LOADER = "HookedTransformer.from_pretrained_no_processing"

DATA_MANIFEST = "prepared_data_manifest.json"

ACTIVATION_MANIFEST = "activation_manifest.json"

RUN_MANIFEST = "saebench_manifest.json"

SAEBENCH_COMMIT = "8042bb3828c6340da8d12062324e92b2077c571c"

def model_provenance():
	return {
		"model_name": MODEL_NAME,
		"hf_model_name": HF_MODEL_NAME,
		"model_revision": MODEL_REVISION,
		"hook_name": HOOK_NAME,
		"hook_layer": HOOK_LAYER,
		"input_dim": INPUT_DIM,
		"loader": LLM_LOADER,
		"llm_dtype": LLM_DTYPE,
	}
