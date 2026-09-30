import math
from dataclasses import dataclass
from numbers import Integral


SAE_MODEL_TYPES = (
	'relu',
	'gated',
	'topk',
	'batch_topk',
	'jumprelu',
	'matryoshka',
)

ALIGNED_MODEL_TYPES = (
	'relu',
	'topk',
	'batch_topk',
)


def default_matryoshka_prefix_sizes(num_latents: int) -> tuple[int, ...]:
	"""Return the paper's five approximately geometric nested widths.

	The standard group fractions are ``[1, 2, 4, 8, 17] / 32``. Their
	cumulative prefix widths are therefore ``[1, 3, 7, 15, 32] / 32``.
	Rounding and de-duplication keep the definition valid for small test models.
	"""
	prefixes = [
		int(num_latents * numerator / 32)
		for numerator in (1, 3, 7, 15)
	]
	prefixes = [size for size in prefixes if 0 < size < num_latents]
	prefixes.append(num_latents)
	return tuple(sorted(set(prefixes)))


@dataclass
class SAEConfig:
	"""Configuration shared by the standalone SAE implementations.

	``num_latents`` is the dictionary width (the ``K=512`` used by BeFOND).
	``k_active`` is the activity budget used by TopK-style encoders.
	"""

	model_type: str = 'topk'
	input_shape: tuple[int, ...] = (1, 16, 16)
	num_latents: int = 512
	k_active: int = 16
	seed: int = 0

	# ReLU and Gated sparsity penalty.
	l1_coefficient: float = 1e-3

	# Dead-latent reconstruction used by TopK-style models.
	auxk_coefficient: float = 1.0 / 32.0
	auxk_k: int | None = None

	# BatchTopK inference threshold moving average.
	threshold_decay: float = 0.999

	# JumpReLU learns one positive threshold per latent.
	jump_threshold: float = 1e-3
	jump_bandwidth: float = 1e-3
	jump_l0_coefficient: float = 1e-3

	# Cumulative prefix widths and loss weights for Matryoshka.
	matryoshka_prefix_sizes: tuple[int, ...] | None = None
	matryoshka_weights: tuple[float, ...] | None = None

	# Parameter-free encoder/decoder alignment reparameterization.
	aligned: bool = False

	def __post_init__(self):
		self.model_type = str(self.model_type).lower()
		self.input_shape = tuple(int(v) for v in self.input_shape)
		for name in ('num_latents', 'k_active', 'seed'):
			value = getattr(self, name)
			if isinstance(value, bool) or not isinstance(value, Integral):
				raise TypeError(f'{name} must be an integer.')
			setattr(self, name, int(value))
		if self.model_type not in SAE_MODEL_TYPES:
			raise ValueError(
				f"Unknown SAE model_type {self.model_type!r}; "
				f"choose from {SAE_MODEL_TYPES}.")
		if not self.input_shape or any(v <= 0 for v in self.input_shape):
			raise ValueError('input_shape must contain positive dimensions.')
		if self.num_latents <= 0:
			raise ValueError('num_latents must be positive.')
		if not 1 <= self.k_active <= self.num_latents:
			raise ValueError(
				'k_active must lie between 1 and num_latents, inclusive.')
		if self.seed < 0:
			raise ValueError('seed must be nonnegative.')
		for name in [
			'l1_coefficient',
			'auxk_coefficient',
			'jump_l0_coefficient',
		]:
			value = getattr(self, name)
			if not math.isfinite(value) or value < 0:
				raise ValueError(f'{name} must be finite and nonnegative.')
		if self.auxk_k is not None:
			if isinstance(self.auxk_k, bool) or not isinstance(
					self.auxk_k, Integral):
				raise TypeError('auxk_k must be an integer when provided.')
			self.auxk_k = int(self.auxk_k)
			if self.auxk_k <= 0:
				raise ValueError('auxk_k must be positive when provided.')
		if (
			not math.isfinite(self.threshold_decay)
			or not 0.0 <= self.threshold_decay < 1.0
		):
			raise ValueError('threshold_decay must lie in [0, 1).')
		if (
			not math.isfinite(self.jump_threshold)
			or not math.isfinite(self.jump_bandwidth)
			or self.jump_threshold <= 0
			or self.jump_bandwidth <= 0
		):
			raise ValueError(
				'jump_threshold and jump_bandwidth must be finite and positive.')
		if self.aligned and self.model_type not in ALIGNED_MODEL_TYPES:
			raise ValueError(
				'aligned training is supported for relu, topk, and '
				'batch_topk models.')

		if self.matryoshka_prefix_sizes is None:
			self.matryoshka_prefix_sizes = default_matryoshka_prefix_sizes(
				self.num_latents)
		else:
			self.matryoshka_prefix_sizes = tuple(
				int(v) for v in self.matryoshka_prefix_sizes)
		if (
			not self.matryoshka_prefix_sizes
			or any(v <= 0 for v in self.matryoshka_prefix_sizes)
			or tuple(sorted(set(self.matryoshka_prefix_sizes)))
			!= self.matryoshka_prefix_sizes
			or self.matryoshka_prefix_sizes[-1] != self.num_latents
		):
			raise ValueError(
				'matryoshka_prefix_sizes must be strictly increasing, '
				'positive, and end at num_latents.')

		if self.matryoshka_weights is not None:
			self.matryoshka_weights = tuple(
				float(v) for v in self.matryoshka_weights)
			if len(self.matryoshka_weights) != len(
				self.matryoshka_prefix_sizes):
				raise ValueError(
					'matryoshka_weights and matryoshka_prefix_sizes must '
					'have equal lengths.')
			if any(v < 0 for v in self.matryoshka_weights):
				raise ValueError('matryoshka_weights must be nonnegative.')
			if not math.isclose(sum(self.matryoshka_weights), 1.0):
				raise ValueError('matryoshka_weights must sum to one.')

	@property
	def input_dim(self) -> int:
		return math.prod(self.input_shape)
