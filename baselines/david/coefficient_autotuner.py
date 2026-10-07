# Verbatim copy of saes/components/coefficient_autotuner.py from
# https://github.com/decoderesearch/synth-sae-bench-experiments @ 66eee052b734193ab2e8cbe6bd9ee6d6caae1675
from dataclasses import dataclass

import torch


@dataclass
class CoefficientAutotunerConfig:
    """Configuration for the CoefficientAutotuner.

    Uses a rate-dampened integral controller:
    1. Tracks current position error (smoothed_l0 - target_l0)
    2. Tracks L0 rate of change
    3. Reduces gain when moving toward target to prevent overshoot

    This is simpler than predictive control and doesn't require tuning a
    prediction horizon.
    """

    target_l0: float
    start_step: int = 0
    # EMA smoothing factor for L0 measurements. Higher = smoother but slower.
    # Effective window ≈ 1/(1-α) steps.
    smoothing_factor: float = 0.99
    # EMA smoothing factor for L0 rate (derivative) estimation.
    rate_smoothing_factor: float = 0.95
    # Integral gain (Ki). Controls speed of multiplier adjustment.
    integral_gain: float = 3e-4
    min_multiplier: float = 1e-2
    max_multiplier: float = 100.0
    # Deadband: no adjustment if |error| <= deadband
    deadband: float = 0.0
    # Nonlinear gain parameter. Adjustment = Ki * tanh(|rel_error| * gain_scale)
    gain_scale: float = 10.0
    # Gain multiplier when error is decreasing (moving toward target).
    # Lower = more damping. 0.01 means 99% reduction when converging.
    convergence_gain: float = 0.01


class CoefficientAutotuner(torch.nn.Module):
    """
    Autotuner that outputs a multiplier to achieve a target L0 sparsity.

    Uses an integral controller with EMA smoothing and gain scheduling. The
    controller reduces its gain when the error is decreasing (i.e., when the
    system is converging toward the target) to prevent overshoot. A tanh
    nonlinearity bounds the adjustment magnitude and provides smooth behavior
    near the setpoint.

    Algorithm:
        1. Smooth L0 measurements using exponential moving average (EMA)
        2. Estimate L0 rate of change (smoothed derivative)
        3. Compute position error: smoothed_l0 - target_l0
        4. Apply gain scheduling: reduce gain when moving toward target
        5. Compute bounded adjustment: Ki * gain * tanh(|rel_error| * scale)
        6. Multiplicative update: multiplier *= (1 ± adjustment)

    References:
        - Integral control and PID controllers:
          Åström, K.J. & Murray, R.M. (2008). "Feedback Systems: An Introduction
          for Scientists and Engineers." Princeton University Press.
          https://fbswiki.org/
        - Gain scheduling:
          Rugh, W.J. & Shamma, J.S. (2000). "Research on gain scheduling."
          Automatica, 36(10), 1401-1425.

    Example usage:
        autotuner = CoefficientAutotuner(cfg)
        for step, batch in enumerate(dataloader):
            feature_acts = sae.encode(batch)
            batch_l0 = (feature_acts != 0).float().sum(dim=-1).mean()
            multiplier = autotuner.update(batch_l0, step)
            effective_coefficient = base_coefficient * multiplier
    """

    _smoothed_l0: torch.Tensor
    _multiplier: torch.Tensor
    _initialized: torch.Tensor
    _l0_rate: torch.Tensor
    _prev_smoothed_l0: torch.Tensor

    def __init__(
        self,
        cfg: CoefficientAutotunerConfig,
        device: torch.device | str = "cpu",
    ):
        """
        Args:
            cfg: Configuration for the autotuner.
            device: Device to store tensors on.
        """
        super().__init__()
        self.cfg = cfg

        self.register_buffer(
            "_smoothed_l0", torch.zeros(1, device=device, dtype=torch.float64)
        )
        self.register_buffer(
            "_multiplier",
            torch.ones(1, device=device, dtype=torch.float64),
        )
        self.register_buffer(
            "_initialized", torch.zeros(1, device=device, dtype=torch.bool)
        )
        # Track rate of L0 change for gain scheduling
        self.register_buffer(
            "_l0_rate", torch.zeros(1, device=device, dtype=torch.float64)
        )
        self.register_buffer(
            "_prev_smoothed_l0", torch.zeros(1, device=device, dtype=torch.float64)
        )

    @property
    def multiplier(self) -> float:
        """Current multiplier value."""
        return self._multiplier.item()

    @property
    def smoothed_l0(self) -> float:
        """Current smoothed L0 estimate."""
        return self._smoothed_l0.item()

    @property
    def l0_rate(self) -> float:
        """Current smoothed rate of L0 change per step."""
        return self._l0_rate.item()

    @torch.no_grad()
    def update(self, batch_l0: float | torch.Tensor, step: int) -> float:
        """
        Update the autotuner state and return the new multiplier.

        Args:
            batch_l0: L0 sparsity from the current batch.
            step: Current training step.

        Returns:
            The updated multiplier value.
        """
        if isinstance(batch_l0, torch.Tensor):
            batch_l0 = batch_l0.item()

        # Update smoothed L0 estimate (EMA)
        if not self._initialized.item():
            self._smoothed_l0.fill_(batch_l0)
            self._prev_smoothed_l0.fill_(batch_l0)
            self._l0_rate.zero_()
            self._initialized.fill_(True)
        else:
            # Store previous smoothed L0 for rate calculation
            self._prev_smoothed_l0.copy_(self._smoothed_l0)

            # EMA: α * old + (1 - α) * new
            self._smoothed_l0.mul_(self.cfg.smoothing_factor).add_(
                batch_l0, alpha=1 - self.cfg.smoothing_factor
            )

            # Update L0 rate (smoothed derivative)
            instant_rate = self._smoothed_l0.item() - self._prev_smoothed_l0.item()
            self._l0_rate.mul_(self.cfg.rate_smoothing_factor).add_(
                instant_rate, alpha=1 - self.cfg.rate_smoothing_factor
            )

        # No adjustment before start_step
        if step < self.cfg.start_step:
            return self.multiplier

        # Position error: positive means L0 is above target
        error = self._smoothed_l0.item() - self.cfg.target_l0

        # Apply deadband - no adjustment if within tolerance
        if abs(error) <= self.cfg.deadband:
            return self.multiplier

        # Determine if we're moving toward or away from target
        # Moving toward target: error and rate have opposite signs
        # (error > 0 and rate < 0) or (error < 0 and rate > 0)
        moving_toward_target = error * self._l0_rate.item() < 0

        # Reduce gain when moving toward target to prevent overshoot
        gain = self.cfg.convergence_gain if moving_toward_target else 1.0

        # Nonlinear gain using tanh (bounded adjustment)
        rel_error = error / self.cfg.target_l0
        adjustment = (
            self.cfg.integral_gain
            * gain
            * torch.tanh(torch.tensor(abs(rel_error) * self.cfg.gain_scale)).item()
        )

        # Multiplicative update
        if error > 0:
            # L0 too high, increase multiplier to make it more sparse
            new_multiplier = self._multiplier.item() * (1 + adjustment)
        else:
            # L0 too low, decrease multiplier to make it less sparse
            new_multiplier = self._multiplier.item() * (1 - adjustment)

        # Clamp to bounds
        new_multiplier = max(
            self.cfg.min_multiplier, min(self.cfg.max_multiplier, new_multiplier)
        )

        self._multiplier.fill_(new_multiplier)
        return self.multiplier

    def reset(self) -> None:
        """Reset smoothed L0 state and multiplier back to 1.0."""
        self._smoothed_l0.zero_()
        self._multiplier.fill_(1.0)
        self._initialized.fill_(False)
        self._l0_rate.zero_()
        self._prev_smoothed_l0.zero_()
