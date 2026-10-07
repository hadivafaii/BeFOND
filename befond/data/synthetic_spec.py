"""Pinned public SynthSAEBench generator and explicit benchmark identity."""

import json
import pathlib
from dataclasses import dataclass
from .files import sha256_file

HIDDEN_DIM = 768
MODEL_ID = "decoderesearch/synth-sae-bench-16k-v1"
MODEL_REVISION = "b2efd8b919ae46d6d487c73d46db5ee52813621d"
SYNTHETIC_SAE_DATA_ROOT = pathlib.Path("data/synthetic")
SYNTHSAEBENCH_DATA_DIR = SYNTHETIC_SAE_DATA_ROOT / MODEL_ID.rsplit("/", 1)[-1]
DEFAULT_GENERATOR_DIR = (
	SYNTHSAEBENCH_DATA_DIR / "generator" / MODEL_REVISION
)
MODEL_NUM_FEATURES = 16_384

PUBLISHED_SOURCE_REPOSITORY = (
	"https://github.com/decoderesearch/synth-sae-bench-experiments"
)
PUBLISHED_SOURCE_REVISION = "66eee052b734193ab2e8cbe6bd9ee6d6caae1675"

SNAPSHOT_FILES = (
	"synthetic_model_config.json",
	"synthetic_model.safetensors",
	"hierarchy.json",
)
SNAPSHOT_HASHES = {
	"synthetic_model_config.json": (
		"ec969226283f05b69fd3b2a8c1cd14b152a998d79a491d732ccd286d096908b5"
	),
	"synthetic_model.safetensors": (
		"4bf0b07f8f6a54a2dfd56ea11bd1ff7c9ca2040b89077446596cf0b375d00560"
	),
	"hierarchy.json": (
		"42b89b24a0496e046c9382e164ced0d59300156f96c49f779c4fba25974ca657"
	),
}


@dataclass(frozen=True)
class BenchmarkSpec:
	"""A fully named SynthSAEBench data distribution."""

	key: str
	storage_name: str
	scale_children_by_parent: bool
	description: str
	model_id: str = MODEL_ID
	model_revision: str = MODEL_REVISION

	@property
	def dataset_name(self):
		return f"SyntheticMechInterp{HIDDEN_DIM}-{self.key}"

	@property
	def generator_dir(self):
		model_dir = (
			SYNTHETIC_SAE_DATA_ROOT / self.model_id.rsplit("/", 1)[-1]
		)
		return model_dir / "generator" / self.model_revision

	@property
	def normalization_dir(self):
		model_dir = (
			SYNTHETIC_SAE_DATA_ROOT / self.model_id.rsplit("/", 1)[-1]
		)
		return model_dir / "normalization" / self.storage_name

	def provenance(self):
		return {
			"benchmark": self.key,
			"model_id": self.model_id,
			"model_revision": self.model_revision,
			"scale_children_by_parent": self.scale_children_by_parent,
			"published_source_repository": PUBLISHED_SOURCE_REPOSITORY,
			"published_source_revision": PUBLISHED_SOURCE_REVISION,
		}


PUBLISHED_BENCHMARK = BenchmarkSpec(
	key="synthsaebench16k-published-scaled",
	storage_name="published-scaled",
	scale_children_by_parent=True,
	description=(
		"Distribution specified by the published experiment source: child "
		"magnitudes are scaled by their active parents."
	),
)
HISTORICAL_FOND_BENCHMARK = BenchmarkSpec(
	key="synthsaebench16k-historical-fond-unscaled",
	storage_name="historical-fond-unscaled",
	scale_children_by_parent=False,
	description=(
		"Pinned distribution used by the completed BeFOND campaign: hierarchy "
		"support is preserved, but child magnitudes are not parent-scaled."
	),
)
BENCHMARKS = {
	PUBLISHED_BENCHMARK.key: PUBLISHED_BENCHMARK,
	HISTORICAL_FOND_BENCHMARK.key: HISTORICAL_FOND_BENCHMARK,
}
BENCHMARK_ALIASES = {
	"published": PUBLISHED_BENCHMARK,
	"historical": HISTORICAL_FOND_BENCHMARK,
}


def resolve_benchmark(benchmark: str | BenchmarkSpec):
	if isinstance(benchmark, BenchmarkSpec):
		return benchmark
	if benchmark in BENCHMARK_ALIASES:
		return BENCHMARK_ALIASES[benchmark]
	try:
		return BENCHMARKS[benchmark]
	except KeyError as error:
		raise ValueError(
			f"Unknown benchmark {benchmark!r}; choose one of {tuple(BENCHMARKS)}."
		) from error


def verify_synthsaebench_snapshot(snapshot, *, hidden_dim=HIDDEN_DIM):
	"""Verify that ``snapshot`` is the pinned public model snapshot."""

	snapshot = pathlib.Path(snapshot).expanduser().resolve()
	hashes = SNAPSHOT_HASHES
	if hidden_dim != HIDDEN_DIM:
		registry = json.loads(pathlib.Path(__file__).with_name("test1_generators.json").read_text())
		hashes = registry[str(hidden_dim)]["files"]
	record_path = snapshot / "test1_generation.json"
	if record_path.exists():
		from experiments.superposition.prepare import generator_config
		record = json.loads(record_path.read_text())
		actual_config = json.loads((snapshot / "synthetic_model_config.json").read_text())
		if record["config"] != generator_config(hidden_dim) or actual_config != record["config"]:
			raise ValueError("Regenerated Test 1 dictionary uses a different generator recipe.")
		hashes = record["files"]
		if set(hashes) != set(SNAPSHOT_FILES) or record["sae_lens"] != "6.49.1":
			raise ValueError("Incomplete or incompatible Test 1 generation record.")
	for name, expected in hashes.items():
		path = snapshot / name
		if not path.is_file():
			raise FileNotFoundError(path)
		actual = sha256_file(path)
		if actual != expected:
			raise ValueError(f"Hash mismatch for {path}: {actual} != {expected}")

	with open(snapshot / "synthetic_model_config.json") as file:
		config = json.load(file)
	if config.get("num_features") != MODEL_NUM_FEATURES:
		raise ValueError("Pinned snapshot has an unexpected feature count.")
	if config.get("hidden_dim") != hidden_dim:
		raise ValueError("Pinned snapshot has an unexpected hidden dimension.")
	return snapshot


def _verify_materialized_snapshot(snapshot):
	snapshot = verify_synthsaebench_snapshot(snapshot)
	for name in SNAPSHOT_FILES:
		path = snapshot / name
		if path.is_symlink():
			raise ValueError(
				f"Canonical SynthSAEBench file must not be a symlink: {path}")
	return snapshot


def download_snapshot(spec, local_dir=None):
	"""Download and verify a materialized copy of the pinned generator."""

	try:
		from huggingface_hub import snapshot_download
	except ModuleNotFoundError as error:
		raise ModuleNotFoundError(
			"Loading SynthSAEBench from Hugging Face requires huggingface_hub."
		) from error
	if local_dir is None:
		local_dir = spec.generator_dir
	snapshot = snapshot_download(
		repo_id=spec.model_id,
		revision=spec.model_revision,
		allow_patterns=list(SNAPSHOT_FILES),
		local_dir=pathlib.Path(local_dir).expanduser(),
	)
	return _verify_materialized_snapshot(snapshot)


def ensure_synthsaebench_snapshot(spec):
	"""Return the canonical generator, downloading it only when absent."""

	try:
		return _verify_materialized_snapshot(spec.generator_dir)
	except FileNotFoundError:
		return download_snapshot(spec)
