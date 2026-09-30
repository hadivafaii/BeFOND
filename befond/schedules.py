import math
from numbers import Integral


LR_SCHEDULER_TYPES = ('cte', 'cos', 'exp', 'lin')


class LRSchedule(object):
	"""Stateless learning-rate schedule indexed by global training step."""

	def __init__(
			self,
			scheduler_type: str,
			lr: float,
			lr_min: float,
			train_steps: int,
			warmup_portion: float = 0.0,
			warm_restart: int = 0,
			warm_restart_peak_mult: float = 1.0,
		):
		if scheduler_type is None:
			scheduler_type = 'cte'
		if scheduler_type not in LR_SCHEDULER_TYPES:
			raise ValueError(
				f"Unknown scheduler type: {scheduler_type!r}.")

		lr = float(lr)
		lr_min = float(lr_min)
		train_steps = int(train_steps)
		warmup_portion = float(warmup_portion)
		warm_restart_peak_mult = float(warm_restart_peak_mult)

		if not 0.0 < lr < float('inf'):
			raise ValueError(f"lr must be finite and positive, got {lr}.")
		if not 0.0 <= lr_min <= lr:
			raise ValueError(
				f"lr_min must satisfy 0 <= lr_min <= lr, got {lr_min}.")
		if train_steps <= 0:
			raise ValueError(
				f"train_steps must be positive, got {train_steps}.")
		if not 0.0 <= warmup_portion < 1.0:
			raise ValueError(
				"warmup_portion must satisfy 0 <= warmup_portion < 1, "
				f"got {warmup_portion}.")
		if scheduler_type == 'exp' and lr_min <= 0.0:
			raise ValueError(
				f"{scheduler_type} scheduling requires lr_min > 0.")
		if isinstance(warm_restart, bool) or not isinstance(warm_restart, Integral) or warm_restart < 0:
			raise ValueError("warm_restart must be a non-negative integer.")
		if warm_restart and scheduler_type not in ('cos', 'exp', 'lin'):
			raise ValueError("warm_restart requires cos, exp, or lin scheduling.")
		if not 0.0 < warm_restart_peak_mult <= 1.0:
			raise ValueError("warm_restart_peak_mult must satisfy 0 < value <= 1.")
		if warm_restart and lr * warm_restart_peak_mult ** warm_restart < lr_min:
			raise ValueError("warm_restart_peak_mult must keep the final peak at least lr_min.")

		self.scheduler_type = scheduler_type
		self.lr = lr
		self.lr_min = lr_min
		self.train_steps = train_steps
		self.warmup_portion = warmup_portion
		self.warm_restart = int(warm_restart)
		self.warm_restart_peak_mult = warm_restart_peak_mult
		self.warmup_steps = train_steps * warmup_portion
		self.decay_start = int(math.ceil(self.warmup_steps))
		self.decay_steps = train_steps - self.decay_start
		if warm_restart and self.decay_steps <= 2 * warm_restart + 1:
			raise ValueError("warm_restart needs at least one step interval per half-cycle after warmup.")

	def _validate_gstep(self, gstep: int) -> int:
		gstep = int(gstep)
		if not 0 <= gstep < self.train_steps:
			raise ValueError(
				f"gstep must satisfy 0 <= gstep < {self.train_steps}, "
				f"got {gstep}.")
		return gstep

	def is_warming_up(self, gstep: int) -> bool:
		gstep = self._validate_gstep(gstep)
		return gstep < self.warmup_steps

	def warmup_factor(self, gstep: int) -> float:
		gstep = self._validate_gstep(gstep)
		if self.warmup_steps == 0 or gstep >= self.warmup_steps:
			return 1.0
		return float(gstep / self.warmup_steps)

	def _decay_progress(self, gstep: int) -> float:
		if self.decay_steps <= 1:
			return 0.0
		return float(
			(gstep - self.decay_start) /
			(self.decay_steps - 1)
		)

	def __call__(self, gstep: int) -> float:
		gstep = self._validate_gstep(gstep)
		if gstep < self.warmup_steps:
			return float(self.lr * self.warmup_factor(gstep))
		if self.scheduler_type == 'cte':
			return self.lr

		progress = self._decay_progress(gstep)
		if progress <= 0.0:
			return self.lr
		if progress >= 1.0:
			return self.lr_min
		# Each restart adds a rise and another fall before the final minimum.
		phase = progress * (2 * self.warm_restart + 1)
		# Change peak at the trough, keeping each rise/fall continuous.
		peak_index = int((phase + 1.0) // 2)
		peak = self.lr * self.warm_restart_peak_mult ** peak_index
		if self.scheduler_type == 'cos':
			coef = 0.5 * (1.0 + math.cos(math.pi * phase))
			return float(
				self.lr_min + coef * (peak - self.lr_min)
			)
		if self.scheduler_type == 'exp':
			if self.warm_restart:
				progress = 1.0 - abs(1.0 - phase % 2.0)
			return float(
				peak * (self.lr_min / peak) ** progress
			)
		if self.scheduler_type == 'lin':
			return float(
				self.lr_min +
				(peak - self.lr_min) * abs(1.0 - phase % 2.0)
			)
		raise RuntimeError(self.scheduler_type)
