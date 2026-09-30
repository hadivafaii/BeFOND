"""Numerical integration of the frozen-statistics decoder NGD flow.

The readable dense solution and scalable matrix-free solutions solve the same
linear ODE: dPhi/dt = (E[x z^T] - Phi E_q[z z^T]) E_metric[z z^T]^{-1}.
The factorized Bernoulli moments are a diagonal plus a low-rank matrix. The
matrix-free paths exploit this structure with Woodbury/CG solves and Taylor or
Chebyshev matrix-function actions. All computations use ordinary PyTorch.
"""

import math

import torch

DECODER_METRICS = (
    "prior", "agg_predictive_prior", "factorized_predictive_prior",
    "agg_posterior", "factorized_posterior",
)
DECODER_SOLVERS = (
    "dense", "matrix_free", "matrix_free_phi1",
    "matrix_free_phi1_chebyshev", "matrix_free_phi1_auto", "capped_quadratic",
)
_MATRIX_FREE_WOODBURY_MAX_RANK = 1024
_MATRIX_FREE_CG_MAX_ITER = 512
_MATRIX_FREE_CG_BLOCK_SIZE = 64
_MATRIX_FREE_TAYLOR_MAX_TERMS = 64
_MATRIX_FREE_MAX_SCALING_STEPS = 1024
_MATRIX_FREE_SPECTRAL_MAX_RANK = 10000
_MATRIX_FREE_SPECTRAL_BLOCK_SIZE = 512
_MATRIX_FREE_CHEBYSHEV_SAFETY_FACTOR = 2.0


def _as_batch_matrix(value):
    return value.unsqueeze(0) if value.ndim == 1 else value



def _new_decoder_solver_diagnostics():
    return {
        'decoder_operator_radius': math.nan,
        'decoder_scaling_steps': 0,
        'decoder_taylor_terms': 0,
        'decoder_operator_applications': 0,
        'decoder_probe_operator_applications': 0,
        'decoder_action_operator_applications': 0,
        'decoder_chebyshev_degree': 0,
        'decoder_chebyshev_spectral_bound': math.nan,
        'decoder_chebyshev_interval_max': math.nan,
        'decoder_chebyshev_coefficient_tail': math.nan,
        'decoder_chebyshev_error_bound': math.nan,
        'decoder_auto_used_chebyshev': 0,
        'decoder_auto_predicted_taylor_steps': 0,
        'decoder_auto_predicted_taylor_applications': 0,
        'decoder_solver_failed': 0,
    }


def _raise_solver_error(message: str, diagnostics: dict):
    diagnostics['decoder_solver_failed'] = 1
    error = RuntimeError(message)
    error.solver_diagnostics = diagnostics.copy()
    raise error


def _validate_fisher_damping(fisher_damping: float) -> float:
    fisher_damping = float(fisher_damping)
    if not math.isfinite(fisher_damping) or fisher_damping < 0.0:
        raise ValueError("Fisher damping must be finite and non-negative.")
    return fisher_damping


class _BatchMomentOperator(object):
    """Diagonal-plus-low-rank batch moment without a dense K x K matrix."""
    def __init__(
            self,
            mean: torch.Tensor,
            variance: torch.Tensor,
            require_positive_diagonal: bool = True,
            diagonal_damping: float = 0.0, ):
        mean = _as_batch_matrix(mean)
        variance = _as_batch_matrix(variance)
        if mean.shape != variance.shape:
            raise ValueError("Mean and variance must have matching shapes.")
        output_dtype = torch.promote_types(mean.dtype, variance.dtype)
        self.diagonal = variance.to(torch.float64).mean(dim=0).to(output_dtype)
        diagonal_damping = _validate_fisher_damping(diagonal_damping)
        if diagonal_damping != 0.0:
            self.diagonal = self.diagonal + diagonal_damping
        if (
                mean.shape[0] > 1
                and torch.equal(mean, mean[:1].expand_as(mean))):
            self.factor_rows = mean[:1]
            self.factor_scale = 1.0
        else:
            self.factor_rows = mean
            self.factor_scale = 1.0 / math.sqrt(mean.shape[0])
        if not torch.isfinite(self.diagonal).all():
            raise ValueError("Moment diagonal must be finite.")
        if require_positive_diagonal:
            if torch.any(self.diagonal <= 0.0):
                raise ValueError("Moment diagonal must be strictly positive.")
        elif torch.any(self.diagonal < 0.0):
            raise ValueError("Moment diagonal must be non-negative.")
        self._woodbury_diagonal_inv = None
        self._woodbury_factor_rows = None
        self._woodbury_scaled_factor_rows = None
        self._woodbury_chol = None
        self._preconditioner = None

    @property
    def size(self):
        return self.diagonal.numel()

    @property
    def rank(self):
        return self.factor_rows.shape[0]

    def matmul(self, rhs: torch.Tensor):
        rhs, was_vector = _as_linear_rhs(rhs, self.size)
        projected = self.factor_rows @ rhs
        low_rank = self.factor_rows.T @ projected
        output = self.diagonal[:, None] * rhs + self.factor_scale ** 2 * low_rank
        return output[:, 0] if was_vector else output

    def solve(self, rhs: torch.Tensor):
        rhs, was_vector = _as_linear_rhs(rhs, self.size)
        if self.rank <= _MATRIX_FREE_WOODBURY_MAX_RANK:
            output = self._woodbury_solve(rhs)
        else:
            blocks = [
                _pcg_solve(self, block)
                for block in rhs.split(_MATRIX_FREE_CG_BLOCK_SIZE, dim=1)
            ]
            output = torch.cat(blocks, dim=1)
        return output[:, 0] if was_vector else output

    def _woodbury_solve(self, rhs: torch.Tensor):
        if self._woodbury_chol is None:
            # The final subtraction can cancel even for a well-conditioned moment.
            # Cache float64 work only for inverses; leave matmul and CG unchanged.
            self._woodbury_diagonal_inv = self.diagonal.to(torch.float64).reciprocal()
            self._woodbury_factor_rows = self.factor_rows.to(torch.float64)
            self._woodbury_scaled_factor_rows = (
                self.factor_scale * self._woodbury_factor_rows)
            weighted_factor_t = (
                self._woodbury_factor_rows.T
                * self._woodbury_diagonal_inv[:, None]
            )
            middle = (
                torch.eye(
                    self.rank,
                    dtype=torch.float64,
                    device=self.diagonal.device,
                )
                + self.factor_scale ** 2
                * (self._woodbury_factor_rows @ weighted_factor_t)
            )
            middle = 0.5 * (middle + middle.T)
            self._woodbury_chol = _checked_cholesky(
                middle, name='Woodbury moment matrix')
        diagonal_inv = self._woodbury_diagonal_inv
        factor_rows = self._woodbury_factor_rows
        diagonal_solution = (
            diagonal_inv[:, None] * rhs.to(torch.float64))
        middle_rhs = self._woodbury_scaled_factor_rows @ diagonal_solution
        middle_solution = torch.cholesky_solve(
            middle_rhs, self._woodbury_chol)
        correction = (
            diagonal_inv[:, None]
            * self.factor_scale
            * (factor_rows.T @ middle_solution)
        )
        return (diagonal_solution - correction).to(rhs.dtype)

    def preconditioner_diagonal(self):
        if self._preconditioner is None:
            self._preconditioner = (
                self.diagonal
                + self.factor_scale ** 2
                * self.factor_rows.square().sum(dim=0)
            )
        return self._preconditioner


class _DecoderMomentMap:
    """Apply E_metric^{-1} E_posterior without constructing either matrix."""

    def __init__(self, posterior, prior):
        self.posterior = posterior
        self.prior = prior

    def __call__(self, rhs):
        return self.prior.solve(self.posterior.matmul(rhs))


def _as_linear_rhs(rhs: torch.Tensor, size: int):
    was_vector = rhs.ndim == 1
    if was_vector:
        rhs = rhs[:, None]
    if rhs.ndim != 2 or rhs.shape[0] != size:
        raise ValueError("Linear-operator right-hand side has invalid shape.")
    return rhs, was_vector


def _matrix_free_rtol(dtype: torch.dtype):
    return 1e-10 if dtype == torch.float64 else 1e-6


def _pcg_solve(operator: _BatchMomentOperator, rhs: torch.Tensor):
    """Jacobi-preconditioned CG with independent vectorized right-hand sides."""
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    rhs_norm = torch.linalg.vector_norm(rhs, dim=0)
    rtol = _matrix_free_rtol(rhs.dtype)
    atol = 10.0 * torch.finfo(rhs.dtype).eps
    tolerance = atol + rtol * rhs_norm
    residual_norm = torch.linalg.vector_norm(r, dim=0)
    if torch.all(residual_norm <= tolerance):
        return x

    preconditioner = operator.preconditioner_diagonal().reciprocal()
    z = preconditioner[:, None] * r
    p = z.clone()
    rz = torch.sum(r * z, dim=0)
    for _ in range(_MATRIX_FREE_CG_MAX_ITER):
        active = residual_norm > tolerance
        p = torch.where(active[None], p, torch.zeros_like(p))
        ap = operator.matmul(p)
        denominator = torch.sum(p * ap, dim=0)
        if torch.any(denominator[active] <= 0.0):
            raise RuntimeError("Matrix-free CG encountered a non-positive direction.")
        alpha = torch.zeros_like(rz)
        alpha[active] = rz[active] / denominator[active]
        x = x + p * alpha[None]
        r = r - ap * alpha[None]
        residual_norm = torch.linalg.vector_norm(r, dim=0)
        if torch.all(residual_norm <= tolerance):
            return x
        z = preconditioner[:, None] * r
        rz_new = torch.sum(r * z, dim=0)
        active_new = residual_norm > tolerance
        beta = torch.zeros_like(rz)
        beta[active_new] = rz_new[active_new] / rz[active_new]
        p = z + p * beta[None]
        rz = rz_new

    max_relative_residual = torch.max(
        residual_norm / torch.clamp(rhs_norm, min=atol)
    ).item()
    raise RuntimeError(
        "Matrix-free CG did not converge after "
        f"{_MATRIX_FREE_CG_MAX_ITER} iterations; maximum relative residual "
        f"was {max_relative_residual:0.3g}."
    )


def _estimate_operator_radius(
        linear_map,
        size,
        dtype,
        device,
        diagnostics,
        n_iterations: int = 8,
        alternating_probe: bool = True, ):
    probe = torch.ones(size, 1, dtype=dtype, device=device)
    if alternating_probe:
        probe[1::2] = -1.0
    probe = probe / torch.linalg.vector_norm(probe)
    estimate = 0.0
    for _ in range(n_iterations):
        mapped = linear_map(probe)
        diagnostics['decoder_operator_applications'] += 1
        diagnostics['decoder_probe_operator_applications'] += 1
        estimate_tensor = torch.linalg.vector_norm(mapped)
        if not torch.isfinite(estimate_tensor):
            _raise_solver_error(
                "Matrix-free NGD operator norm estimate is not finite.",
                diagnostics,
            )
        estimate = estimate_tensor.item()
        if estimate == 0.0:
            return 0.0
        probe = mapped / estimate_tensor
    return estimate


def _taylor_exponential_action(
        linear_map,
        rhs: torch.Tensor,
        scale: float,
        diagnostics: dict, ):
    """Apply exp(-scale * A) using scaled operator-only Taylor actions."""
    if scale == 0.0 or not torch.any(rhs):
        return rhs.clone()
    rtol = _matrix_free_rtol(rhs.dtype)
    radius = _estimate_operator_radius(
        linear_map=linear_map,
        size=rhs.shape[0],
        dtype=rhs.dtype,
        device=rhs.device,
        diagnostics=diagnostics,
    )
    n_steps = max(1, math.ceil(abs(float(scale)) * radius / 0.5))
    diagnostics['decoder_operator_radius'] = radius
    diagnostics['decoder_scaling_steps'] = n_steps
    if n_steps > _MATRIX_FREE_MAX_SCALING_STEPS:
        _raise_solver_error(
            "Matrix-free exponential action requires too many scaling steps: "
            f"{n_steps} > {_MATRIX_FREE_MAX_SCALING_STEPS}.",
            diagnostics,
        )
    step_scale = -float(scale) / n_steps
    output = rhs.clone()
    step_rtol = max(
        10.0 * torch.finfo(rhs.dtype).eps,
        rtol / n_steps,
    )
    for _ in range(n_steps):
        term = output
        step_output = output.clone()
        converged = False
        for degree in range(1, _MATRIX_FREE_TAYLOR_MAX_TERMS + 1):
            term = linear_map(term) * (step_scale / degree)
            diagnostics['decoder_taylor_terms'] += 1
            diagnostics['decoder_operator_applications'] += 1
            diagnostics['decoder_action_operator_applications'] += 1
            step_output = step_output + term
            term_norm = torch.linalg.vector_norm(term)
            output_norm = torch.linalg.vector_norm(step_output)
            if term_norm <= step_rtol * torch.clamp(
                    output_norm, min=torch.finfo(rhs.dtype).tiny):
                converged = True
                break
        if not converged:
            _raise_solver_error(
                "Matrix-free exponential action did not converge within "
                f"{_MATRIX_FREE_TAYLOR_MAX_TERMS} Taylor terms.",
                diagnostics,
            )
        output = step_output
    return output


def _chebyshev_phi1_coefficients(
        interval_max: float,
        degree: int,
        dtype: torch.dtype,
        device: torch.device, ):
    """Interpolate (1 - exp(-x)) / x on [0, interval_max]."""
    n_nodes = degree + 1
    work_dtype = torch.float64
    indices = torch.arange(n_nodes, dtype=work_dtype, device=device)
    theta = math.pi * (indices + 0.5) / n_nodes
    nodes = torch.cos(theta)
    arguments = 0.5 * float(interval_max) * (nodes + 1.0)
    values = torch.where(
        arguments == 0.0,
        torch.ones_like(arguments),
        -torch.expm1(-arguments) / arguments,
    )
    basis = torch.cos(indices[:, None] * theta[None])
    coefficients = (2.0 / n_nodes) * (basis @ values)
    coefficients[0] *= 0.5
    return coefficients.to(dtype)


def _bound_decoder_operator_radius(
        posterior: _BatchMomentOperator,
        prior: _BatchMomentOperator,
        diagnostics: dict, ) -> float:
    """Bound the spectrum of E_prior^-1 E_post without fixed-vector probes."""
    work_dtype = (
        torch.float64 if prior.diagonal.dtype == torch.float64
        else torch.float32)
    prior_diagonal = prior.diagonal.to(work_dtype)
    diagonal_bound = (posterior.diagonal / prior_diagonal).max()
    inv_sqrt = prior_diagonal.rsqrt()
    prior_mean = prior.factor_rows.to(work_dtype).mean(dim=0) * inv_sqrt
    factor = posterior.factor_rows.to(work_dtype) * inv_sqrt

    # Jensen: E_prior >= D + mean(prior) mean(prior)^T, including damping.
    # Whiten this lower bound; its diagonal contribution is bounded above
    # by max(E_post.diagonal / D), since (I + vv^T)^-1/2 is a contraction.
    root = torch.sqrt(1.0 + prior_mean.square().sum())
    factor = factor - (
        (factor @ prior_mean) / (root * (root + 1.0))
    )[:, None] * prior_mean
    scale = posterior.factor_scale ** 2
    low_rank_bound = factor.square().sum(dtype=torch.float64) * scale
    if posterior.rank <= _MATRIX_FREE_SPECTRAL_MAX_RANK:
        # Exact Gram row sums remain valid when the batch exceeds K.
        # Tile the rows to avoid retaining a full batch-space Gram matrix.
        row_bound = torch.zeros_like(low_rank_bound)
        for block in factor.split(_MATRIX_FREE_SPECTRAL_BLOCK_SIZE):
            gram = block @ factor.T
            block_bound = gram.abs().sum(dim=1, dtype=torch.float64).max()
            row_bound = torch.maximum(row_bound, block_bound)
        low_rank_bound = torch.minimum(low_rank_bound, row_bound * scale)
    else:
        # |F F.T| <= |F| |F|.T gives a conservative row-sum bound in O(BK).
        # Avoid the large batch Gram matrix and the trace-only overestimate.
        absolute = factor.abs()
        row_bound = (absolute @ absolute.sum(dim=0)).max().to(torch.float64) * scale
        low_rank_bound = torch.minimum(low_rank_bound, row_bound)
    radius = (diagonal_bound + low_rank_bound).item() * (1.0 + 1e-5)
    if not math.isfinite(radius):
        _raise_solver_error(
            "Matrix-free NGD spectral bound is not finite.", diagnostics)
    return radius


def _predict_taylor_affine_work(
        radius: float,
        scale: float,
        dtype: torch.dtype, ) -> tuple[int, int]:
    n_steps = max(1, math.ceil(abs(float(scale)) * radius / 0.5))
    step_radius = abs(float(scale)) * radius / n_steps
    step_rtol = max(
        10.0 * torch.finfo(dtype).eps,
        _matrix_free_rtol(dtype) / n_steps,
    )
    term_bound = step_radius
    n_terms = _MATRIX_FREE_TAYLOR_MAX_TERMS
    for degree in range(1, _MATRIX_FREE_TAYLOR_MAX_TERMS + 1):
        term_bound *= step_radius / (degree + 1)
        if term_bound <= step_rtol:
            n_terms = degree
            break
    return n_steps, n_steps * (1 + n_terms)


def _adaptive_chebyshev_phi1_coefficients(
        interval_max: float,
        tolerance: float,
        max_degree: int,
        dtype: torch.dtype,
        device: torch.device,
        diagnostics: dict, ) -> tuple[torch.Tensor, int]:
    max_probe_degree = max(2, 2 * max_degree)
    probe_degree = min(16, max_probe_degree)
    if device.type == 'cuda':
        # A single larger probe avoids repeated GPU launches and synchronization.
        probe_degree = min(max(32, max_degree), max_probe_degree)
    while True:
        coefficients = _chebyshev_phi1_coefficients(
            interval_max=interval_max,
            degree=probe_degree,
            dtype=torch.float64,
            device=device,
        )
        unresolved_tail = coefficients[
            -min(4, len(coefficients)):
        ].abs().sum()
        if unresolved_tail <= tolerance:
            abs_coefficients = coefficients.abs()
            omitted_mass = torch.zeros_like(abs_coefficients)
            omitted_mass[:-1] = torch.flip(torch.cumsum(
                torch.flip(abs_coefficients[1:], dims=(0,)), dim=0),
                dims=(0,),
            )
            candidate_limit = min(max_degree, probe_degree // 2)
            candidates = torch.nonzero(
                omitted_mass[1:candidate_limit + 1] <= float(tolerance),
                as_tuple=False,
            )
            if len(candidates) > 0:
                degree = int(candidates[0].item()) + 1
                diagnostics['decoder_chebyshev_error_bound'] = (
                    omitted_mass[degree].item())
                return coefficients[:degree + 1].to(dtype), degree
        if probe_degree == max_probe_degree:
            if unresolved_tail > tolerance:
                message = (
                    "Automatic Chebyshev phi_1 action could not resolve its "
                    f"coefficient tail through probe degree {probe_degree}; "
                    "terminal coefficient mass is "
                    f"{unresolved_tail.item():0.3g}.")
            else:
                message = (
                    "Automatic Chebyshev phi_1 action needs a degree above "
                    f"{max_degree} to satisfy tolerance {tolerance:0.3g}.")
            _raise_solver_error(message, diagnostics)
        probe_degree = min(2 * probe_degree, max_probe_degree)


def _chebyshev_affine_flow_action(
        linear_map,
        state: torch.Tensor,
        drive: torch.Tensor,
        scale: float,
        degree: int,
        diagnostics: dict,
        radius: float,
        coefficients: torch.Tensor = None, ):
    """Integrate y' = drive - A y with a fixed-degree phi_1 action."""
    diagnostics['decoder_chebyshev_degree'] = degree
    if scale == 0.0:
        return state.clone()
    if scale < 0.0:
        raise ValueError("Chebyshev phi_1 action requires a non-negative scale.")
    diagnostics['decoder_operator_radius'] = radius
    if radius == 0.0:
        diagnostics['decoder_chebyshev_spectral_bound'] = 0.0
        diagnostics['decoder_chebyshev_interval_max'] = 0.0
        diagnostics['decoder_chebyshev_degree'] = 0
        return state + float(scale) * drive

    spectral_bound = _MATRIX_FREE_CHEBYSHEV_SAFETY_FACTOR * radius
    interval_max = float(scale) * spectral_bound
    if coefficients is None:
        coefficients = _chebyshev_phi1_coefficients(
            interval_max=interval_max,
            degree=degree,
            dtype=state.dtype,
            device=state.device,
        )
    else:
        coefficients = coefficients.to(dtype=state.dtype, device=state.device)
        degree = len(coefficients) - 1
        diagnostics['decoder_chebyshev_degree'] = degree
    diagnostics['decoder_chebyshev_spectral_bound'] = spectral_bound
    diagnostics['decoder_chebyshev_interval_max'] = interval_max
    tail_start = max(0, degree - 3)
    diagnostics['decoder_chebyshev_coefficient_tail'] = (
        coefficients[tail_start:].abs().sum()
        / torch.clamp(
            coefficients.abs().sum(),
            min=torch.finfo(coefficients.dtype).tiny,
        )
    ).item()

    residual = drive - linear_map(state)
    diagnostics['decoder_operator_applications'] += 1
    diagnostics['decoder_action_operator_applications'] += 1
    term_previous = residual
    output = coefficients[0] * term_previous
    if degree >= 1:
        term_current = (
            2.0 * linear_map(term_previous) / spectral_bound
            - term_previous
        )
        diagnostics['decoder_operator_applications'] += 1
        diagnostics['decoder_action_operator_applications'] += 1
        output = output + coefficients[1] * term_current
        for index in range(2, degree + 1):
            term_next = (
                4.0 * linear_map(term_current) / spectral_bound
                - 2.0 * term_current - term_previous
            )
            output = output + coefficients[index] * term_next
            diagnostics['decoder_operator_applications'] += 1
            diagnostics['decoder_action_operator_applications'] += 1
            term_previous, term_current = term_current, term_next
    updated = state + float(scale) * output
    if not torch.isfinite(updated).all():
        _raise_solver_error(
            "Matrix-free Chebyshev phi_1 action is not finite.",
            diagnostics,
        )
    return updated


def _auto_affine_flow_action(
        linear_map,
        state: torch.Tensor,
        drive: torch.Tensor,
        scale: float,
        taylor_max_applications: int,
        chebyshev_tolerance: float,
        chebyshev_max_degree: int,
        diagnostics: dict,
        radius: float, ):
    """Select a bounded-work Taylor or adaptive Chebyshev phi_1 action."""
    if scale == 0.0:
        return state.clone()
    if scale < 0.0:
        raise ValueError("Automatic phi_1 action requires a non-negative scale.")
    diagnostics['decoder_operator_radius'] = radius
    n_steps, predicted_applications = _predict_taylor_affine_work(
        radius=radius,
        scale=scale,
        dtype=state.dtype,
    )
    diagnostics['decoder_auto_predicted_taylor_steps'] = n_steps
    diagnostics['decoder_auto_predicted_taylor_applications'] = (
        predicted_applications)
    use_taylor = (
        n_steps <= _MATRIX_FREE_MAX_SCALING_STEPS
        and predicted_applications <= taylor_max_applications
    )
    if use_taylor:
        return _taylor_affine_flow_action(
            linear_map=linear_map,
            state=state,
            drive=drive,
            scale=scale,
            diagnostics=diagnostics,
            radius=radius,
        )

    diagnostics['decoder_auto_used_chebyshev'] = 1
    if radius == 0.0:
        return state + float(scale) * drive
    interval_max = (
        float(scale) * _MATRIX_FREE_CHEBYSHEV_SAFETY_FACTOR * radius)
    coefficients, degree = _adaptive_chebyshev_phi1_coefficients(
        interval_max=interval_max,
        tolerance=chebyshev_tolerance,
        max_degree=chebyshev_max_degree,
        dtype=state.dtype,
        device=state.device,
        diagnostics=diagnostics,
    )
    return _chebyshev_affine_flow_action(
        linear_map=linear_map,
        state=state,
        drive=drive,
        scale=scale,
        degree=degree,
        diagnostics=diagnostics,
        radius=radius,
        coefficients=coefficients,
    )


def _taylor_affine_flow_action(
        linear_map,
        state: torch.Tensor,
        drive: torch.Tensor,
        scale: float,
        diagnostics: dict,
        radius: float = None, ):
    """Integrate y' = drive - A y using inverse-free phi_1 actions."""
    if scale == 0.0:
        return state.clone()
    rtol = _matrix_free_rtol(state.dtype)
    if radius is None:
        radius = _estimate_operator_radius(
            linear_map=linear_map,
            size=state.shape[0],
            dtype=state.dtype,
            device=state.device,
            diagnostics=diagnostics,
        )
    n_steps = max(1, math.ceil(abs(float(scale)) * radius / 0.5))
    diagnostics['decoder_operator_radius'] = radius
    diagnostics['decoder_scaling_steps'] = n_steps
    if n_steps > _MATRIX_FREE_MAX_SCALING_STEPS:
        _raise_solver_error(
            "Matrix-free phi_1 action requires too many scaling steps: "
            f"{n_steps} > {_MATRIX_FREE_MAX_SCALING_STEPS}.",
            diagnostics,
        )
    step_scale = float(scale) / n_steps
    step_rtol = max(
        10.0 * torch.finfo(state.dtype).eps,
        rtol / n_steps,
    )
    output = state.clone()
    for _ in range(n_steps):
        residual = drive - linear_map(output)
        diagnostics['decoder_operator_applications'] += 1
        diagnostics['decoder_action_operator_applications'] += 1
        term = residual * step_scale
        step_output = output + term
        converged = False
        for degree in range(1, _MATRIX_FREE_TAYLOR_MAX_TERMS + 1):
            term = linear_map(term) * (-step_scale / (degree + 1))
            diagnostics['decoder_taylor_terms'] += 1
            diagnostics['decoder_operator_applications'] += 1
            diagnostics['decoder_action_operator_applications'] += 1
            step_output = step_output + term
            term_norm = torch.linalg.vector_norm(term)
            output_norm = torch.linalg.vector_norm(step_output)
            if term_norm <= step_rtol * torch.clamp(
                    output_norm, min=torch.finfo(state.dtype).tiny):
                converged = True
                break
        if not converged:
            _raise_solver_error(
                "Matrix-free phi_1 action did not converge within "
                f"{_MATRIX_FREE_TAYLOR_MAX_TERMS} Taylor terms.",
                diagnostics,
            )
        output = step_output
    return output


def _batch_decoder_ngd_update_matrix_free(
        phi: torch.Tensor,
        x: torch.Tensor,
        mean: torch.Tensor,
        prior_mean: torch.Tensor,
        variance: torch.Tensor,
        prior_variance: torch.Tensor,
        eta: float,
        delta_t: float,
        fisher_damping: float,
        diagnostics: dict,
        prior_moment=None, ):
    posterior_moment = _BatchMomentOperator(mean, variance)
    if prior_moment is None:
        prior_moment = _BatchMomentOperator(
            prior_mean, prior_variance, diagonal_damping=fisher_damping)
    xr = torch.einsum('bm,bk->mk', x, mean) / x.shape[0]
    phi_star = posterior_moment.solve(xr.T).T
    delta = (phi - phi_star).T

    decay_operator = _DecoderMomentMap(posterior_moment, prior_moment)

    delta_decayed = _taylor_exponential_action(
        linear_map=decay_operator,
        rhs=delta,
        scale=float(eta) * float(delta_t),
        diagnostics=diagnostics,
    )
    return phi_star + delta_decayed.T


def _batch_decoder_ngd_update_matrix_free_phi1(
        phi: torch.Tensor,
        x: torch.Tensor,
        mean: torch.Tensor,
        prior_mean: torch.Tensor,
        variance: torch.Tensor,
        prior_variance: torch.Tensor,
        eta: float,
        delta_t: float,
        fisher_damping: float,
        diagnostics: dict,
        prior_moment=None, ):
    posterior_moment = _BatchMomentOperator(
        mean=mean,
        variance=variance,
        require_positive_diagonal=False,
    )
    if prior_moment is None:
        prior_moment = _BatchMomentOperator(
            prior_mean, prior_variance, diagonal_damping=fisher_damping)
    xr = torch.einsum('bm,bk->mk', x, mean) / x.shape[0]
    state = phi.T
    drive = prior_moment.solve(xr.T)

    decay_operator = _DecoderMomentMap(posterior_moment, prior_moment)

    updated = _taylor_affine_flow_action(
        linear_map=decay_operator,
        state=state,
        drive=drive,
        scale=float(eta) * float(delta_t),
        diagnostics=diagnostics,
    )
    return updated.T


def _batch_decoder_ngd_update_matrix_free_phi1_chebyshev(
        phi: torch.Tensor,
        x: torch.Tensor,
        mean: torch.Tensor,
        prior_mean: torch.Tensor,
        variance: torch.Tensor,
        prior_variance: torch.Tensor,
        eta: float,
        delta_t: float,
        fisher_damping: float,
        chebyshev_degree: int,
        diagnostics: dict,
        prior_moment=None, ):
    posterior_moment = _BatchMomentOperator(
        mean=mean,
        variance=variance,
        require_positive_diagonal=False,
    )
    if prior_moment is None:
        prior_moment = _BatchMomentOperator(
            prior_mean, prior_variance, diagonal_damping=fisher_damping)
    xr = torch.einsum('bm,bk->mk', x, mean) / x.shape[0]
    state = phi.T
    drive = prior_moment.solve(xr.T)

    decay_operator = _DecoderMomentMap(posterior_moment, prior_moment)

    updated = _chebyshev_affine_flow_action(
        linear_map=decay_operator,
        state=state,
        drive=drive,
        scale=float(eta) * float(delta_t),
        degree=chebyshev_degree,
        diagnostics=diagnostics,
        radius=_bound_decoder_operator_radius(
            posterior_moment, prior_moment, diagnostics,
        ) if eta != 0.0 and delta_t != 0.0 else 0.0,
    )
    return updated.T


def _batch_decoder_ngd_update_matrix_free_phi1_auto(
        phi: torch.Tensor,
        x: torch.Tensor,
        mean: torch.Tensor,
        prior_mean: torch.Tensor,
        variance: torch.Tensor,
        prior_variance: torch.Tensor,
        eta: float,
        delta_t: float,
        fisher_damping: float,
        taylor_max_applications: int,
        chebyshev_tolerance: float,
        chebyshev_max_degree: int,
        diagnostics: dict,
        prior_moment=None, ):
    posterior_moment = _BatchMomentOperator(
        mean=mean,
        variance=variance,
        require_positive_diagonal=False,
    )
    if prior_moment is None:
        prior_moment = _BatchMomentOperator(
            prior_mean, prior_variance, diagonal_damping=fisher_damping)
    xr = torch.einsum('bm,bk->mk', x, mean) / x.shape[0]
    state = phi.T
    drive = prior_moment.solve(xr.T)

    decay_operator = _DecoderMomentMap(posterior_moment, prior_moment)

    updated = _auto_affine_flow_action(
        linear_map=decay_operator,
        state=state,
        drive=drive,
        scale=float(eta) * float(delta_t),
        taylor_max_applications=taylor_max_applications,
        chebyshev_tolerance=chebyshev_tolerance,
        chebyshev_max_degree=chebyshev_max_degree,
        diagnostics=diagnostics,
        radius=_bound_decoder_operator_radius(
            posterior_moment, prior_moment, diagnostics,
        ) if eta != 0.0 and delta_t != 0.0 else 0.0,
    )
    return updated.T


def second_moment(
        mean: torch.Tensor,
        variance: torch.Tensor, ):
    mean = mean.reshape(-1)
    variance = variance.reshape(-1)
    if mean.shape != variance.shape:
        raise ValueError("Mean and variance must have matching shapes.")
    return torch.diag(variance) + torch.outer(mean, mean)


def _batch_mean_second_moment_float64(
        mean: torch.Tensor,
        variance: torch.Tensor, ):
    batch_size = mean.shape[0]
    mean_work = mean.to(torch.float64)
    variance_work = variance.to(torch.float64)
    return (
        mean_work.T @ mean_work / batch_size
        + torch.diag(variance_work.mean(dim=0))
    )


def batch_mean_second_moment(
        mean: torch.Tensor,
        variance: torch.Tensor, ):
    """Return the batch mean of the per-sample second moments."""
    mean = _as_batch_matrix(mean)
    variance = _as_batch_matrix(variance)
    if mean.shape != variance.shape:
        raise ValueError("Mean and variance must have matching shapes.")
    output_dtype = torch.promote_types(mean.dtype, variance.dtype)
    return _batch_mean_second_moment_float64(
        mean=mean,
        variance=variance,
    ).to(output_dtype)


def _factorized_moments(mean, variance):
    """Product-of-marginals projection, including between-example variance."""
    dtype = torch.promote_types(mean.dtype, variance.dtype)
    mean_work = mean.to(torch.float64)
    projected_mean = mean_work.mean(dim=0, keepdim=True)
    projected_variance = (
        variance.to(torch.float64).mean(dim=0, keepdim=True)
        + mean_work.var(dim=0, correction=0, keepdim=True)
    )
    return projected_mean.to(dtype), projected_variance.to(dtype)


def _batch_decoder_posterior_update(
        phi: torch.Tensor,
        x: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        eta: float,
        delta_t: float,
        decoder_solver: str,
        fisher_damping: float, ):
    """Exact posterior relaxation, or a damped step when damping is positive."""
    scale = float(eta) * float(delta_t)
    if scale == 0.0:
        return phi.clone()
    output_dtype = phi.dtype
    if decoder_solver == 'dense':
        phi, x, mean, variance = (
            tensor.to(torch.float64) for tensor in (phi, x, mean, variance))
    if fisher_damping == 0.0:
        rhs = x.T @ mean / x.shape[0]
    else:
        # Keep the objective's moments undamped; damping changes only mobility.
        residual = x - mean @ phi.T
        mean_variance = variance.to(torch.float64).mean(dim=0).to(phi.dtype)
        rhs = residual.T @ mean / x.shape[0] - phi * mean_variance
    if decoder_solver == 'dense':
        moment = _batch_mean_second_moment_float64(mean, variance)
        moment.diagonal().add_(fisher_damping)
        chol = _checked_cholesky(moment, name='Posterior metric matrix')
        direction = torch.cholesky_solve(rhs.T, chol).T
    else:
        moment = _BatchMomentOperator(
            mean, variance, require_positive_diagonal=False,
            diagonal_damping=fisher_damping)
        if torch.any(moment.diagonal <= 0.0):
            raise ValueError(
                "Posterior metric diagonal must be strictly positive; "
                "use fisher_damping > 0 for zero posterior variances.")
        if moment.rank > _MATRIX_FREE_WOODBURY_MAX_RANK:
            # Tiny rates can give a tiny RHS but an order-one regression target.
            rhs_scale = rhs.abs().amax(dim=1, keepdim=True)
            rhs_scale = torch.where(rhs_scale > 0.0, rhs_scale, 1.0)
            direction = moment.solve((rhs / rhs_scale).T).T * rhs_scale
        else:
            direction = moment.solve(rhs.T).T
    if fisher_damping == 0.0:
        direction = direction - phi
    return (phi - math.expm1(-scale) * direction).to(output_dtype)


def _batch_decoder_capped_quadratic_update(
        phi, x, mean, variance, prior_mean, prior_variance, eta, delta_t,
        fisher_damping, observation_precision, diagnostics, reduce_stats=None):
    """Rank-one mobility and a line minimum capped by the scheduled flow time.

    The objective fixes posterior moments, bias and observation variance, before
    decoder norm regularization. Output shards must sum their line statistics.
    """
    cap = float(eta) * float(delta_t)
    if not math.isfinite(cap) or cap < 0:
        raise ValueError('Capped quadratic step cap must be finite and nonnegative.')
    p, x, mu, var = (value.to(torch.float64) for value in (phi, x, mean, variance))
    r, v = (value.to(torch.float64).reshape(-1) for value in (prior_mean, prior_variance))
    d = v + fisher_damping
    if not bool(torch.isfinite(d).all() & (d > 0).all()):
        _raise_solver_error('Capped quadratic requires a finite positive metric diagonal.', diagnostics)
    precision = (p.new_ones(len(p)) if observation_precision is None else
        observation_precision.to(dtype=p.dtype, device=p.device).reshape(-1).expand(len(p)))
    vbar = var.mean(dim=0)
    gradient = (x - mu @ p.T).T @ mu / len(x) - p * vbar
    w = r / d
    direction = gradient / d - ((gradient @ w) / (1 + r @ w))[:, None] * w
    projection = mu @ direction.T
    slope = (precision[:, None] * gradient * direction).sum()
    curvature = ((projection.square() * precision).sum() / len(x)
        + (precision[:, None] * direction.square() * vbar).sum())
    valid = ((precision > 0).all() & (var >= 0).all()
        & torch.isfinite(direction).all() & torch.isfinite(precision).all())
    # Include shard-local failures in the collective so every rank fails together.
    stats = torch.stack((slope, curvature, (~valid).to(p.dtype)))
    if reduce_stats is not None:
        reduce_stats(stats)
    slope, curvature, invalid = stats.unbind()
    if not bool(torch.isfinite(stats).all()) or bool(invalid > 0):
        _raise_solver_error('Invalid capped quadratic inputs or line statistics.', diagnostics)
    if bool((slope < 0) | (curvature < 0) | ((slope > 0) & (curvature == 0))):
        _raise_solver_error('Capped quadratic requires a descending direction and positive curvature.', diagnostics)
    alpha = slope.new_zeros(()) if bool(slope == 0) else (slope / curvature).clamp(max=cap)
    output = (p + alpha * direction).to(phi.dtype)
    # The shared alpha is valid, but a cast can still overflow on only one shard.
    invalid_output = (~torch.isfinite(output).all()).to(p.dtype)
    if reduce_stats is not None:
        reduce_stats(invalid_output)
    if bool(invalid_output > 0):
        _raise_solver_error('Nonfinite capped quadratic decoder proposal.', diagnostics)
    diagnostics.update(
        decoder_quadratic_alpha=float(alpha), decoder_quadratic_cap=cap,
        decoder_quadratic_limited=int(alpha < cap),
        decoder_quadratic_slope=float(slope), decoder_quadratic_curvature=float(curvature),
        decoder_quadratic_change=float(-alpha * slope + 0.5 * alpha.square() * curvature))
    return output


def _checked_cholesky(matrix: torch.Tensor, name: str):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    if not torch.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")
    chol, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
    if int(info.item()) != 0:
        raise ValueError(
            f"{name} must be positive definite; Cholesky factorization failed."
        )
    return chol


def _dense_decoder_update(phi, x, mean, variance, metric_mean, metric_variance,
                          scale, fisher_damping):
    """Solve the frozen flow by diagonalizing a symmetric whitened moment."""
    dtype = phi.dtype
    phi = phi.double()
    cross = x.double().T @ mean.double() / len(x)
    posterior = _batch_mean_second_moment_float64(mean, variance)
    metric = _batch_mean_second_moment_float64(metric_mean, metric_variance)
    metric.diagonal().add_(fisher_damping)
    chol_posterior = _checked_cholesky(posterior, "Posterior moment matrix")
    chol_metric = _checked_cholesky(metric, "Metric moment matrix")
    equilibrium = torch.cholesky_solve(cross.T, chol_posterior).T
    left = torch.linalg.solve_triangular(chol_metric, posterior, upper=False)
    whitened = torch.linalg.solve_triangular(chol_metric, left.T, upper=False).T
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    if not torch.isfinite(eigenvalues).all() or torch.any(eigenvalues <= 0):
        raise ValueError("Whitened decoder moment must be positive definite.")
    decay = (eigenvectors * torch.exp(-scale * eigenvalues)[None]) @ eigenvectors.T
    delta = (phi - equilibrium) @ chol_metric @ decay
    delta = torch.linalg.solve_triangular(chol_metric.T, delta.T, upper=True).T
    return (equilibrium + delta).to(dtype)


@torch.no_grad()
def decoder_update(*, phi, x, mean, variance, prior_mean, prior_variance,
                   learning_rate, delta_t=1.0, decoder_metric="prior",
                   decoder_solver="dense", fisher_damping=0.0,
                   chebyshev_degree=128, phi1_auto_taylor_max_applications=32,
                   phi1_auto_chebyshev_tolerance=1e-5,
                   phi1_auto_chebyshev_max_degree=512,
                   observation_precision=None, _reduce_line_statistics=None):
    """Return the updated [D,K] decoder and numerical diagnostics.

    ``prior_mean``/``prior_variance`` describe the selected metric: one shared
    baseline prior row for ``prior``, or the batch's predictive references for
    the predictive-prior metrics. The posterior metrics use ``mean``/``variance``.
    These are Bernoulli moments; both variances must be supplied explicitly.
    """
    if decoder_metric not in DECODER_METRICS:
        raise ValueError(f"Unknown decoder metric: {decoder_metric!r}.")
    if decoder_solver not in DECODER_SOLVERS:
        raise ValueError(f"Unknown decoder solver: {decoder_solver!r}.")
    fisher_damping = _validate_fisher_damping(fisher_damping)
    scale = float(learning_rate) * float(delta_t)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Decoder flow time must be finite and nonnegative.")
    if chebyshev_degree < 1 or phi1_auto_chebyshev_max_degree < 1:
        raise ValueError("Chebyshev degrees must be positive.")
    if phi1_auto_taylor_max_applications < 0 or phi1_auto_chebyshev_tolerance <= 0:
        raise ValueError("Invalid automatic solver work limit or tolerance.")
    prior_mean, prior_variance = map(_as_batch_matrix, (prior_mean, prior_variance))
    if x.ndim != 2 or phi.ndim != 2 or mean.ndim != 2:
        raise ValueError("Decoder, observations and posterior means must be matrices.")
    if (phi.shape != (x.shape[1], mean.shape[1]) or len(x) != len(mean)
            or variance.shape != mean.shape or prior_mean.shape != prior_variance.shape
            or prior_mean.shape[1] != mean.shape[1]):
        raise ValueError("Incompatible decoder or moment shapes.")
    if decoder_metric == "prior" and len(prior_mean) != 1:
        raise ValueError("The prior metric requires one shared baseline-prior row.")
    if decoder_metric.startswith("factorized_"):
        source = ((mean, variance) if decoder_metric == "factorized_posterior"
                  else (prior_mean, prior_variance))
        prior_mean, prior_variance = _factorized_moments(*source)
    diagnostics = _new_decoder_solver_diagnostics()
    if scale == 0:
        return phi.clone(), diagnostics
    if decoder_solver == "capped_quadratic":
        if decoder_metric not in ("prior", "factorized_predictive_prior", "factorized_posterior"):
            raise ValueError("Capped quadratic requires a factorized metric.")
        output = _batch_decoder_capped_quadratic_update(
            phi, x, mean, variance, prior_mean, prior_variance, learning_rate, delta_t,
            fisher_damping, observation_precision, diagnostics, _reduce_line_statistics)
    elif decoder_metric == "agg_posterior":
        output = _batch_decoder_posterior_update(
            phi, x, mean, variance, learning_rate, delta_t, decoder_solver, fisher_damping)
    elif decoder_solver == "dense":
        output = _dense_decoder_update(phi, x, mean, variance, prior_mean,
                                      prior_variance, scale, fisher_damping)
    else:
        arguments = dict(phi=phi, x=x, mean=mean, variance=variance,
                         prior_mean=prior_mean, prior_variance=prior_variance,
                         eta=learning_rate, delta_t=delta_t,
                         fisher_damping=fisher_damping, diagnostics=diagnostics)
        if decoder_solver == "matrix_free":
            output = _batch_decoder_ngd_update_matrix_free(**arguments)
        elif decoder_solver == "matrix_free_phi1":
            output = _batch_decoder_ngd_update_matrix_free_phi1(**arguments)
        elif decoder_solver == "matrix_free_phi1_chebyshev":
            output = _batch_decoder_ngd_update_matrix_free_phi1_chebyshev(
                **arguments, chebyshev_degree=chebyshev_degree)
        else:
            output = _batch_decoder_ngd_update_matrix_free_phi1_auto(
                **arguments, taylor_max_applications=phi1_auto_taylor_max_applications,
                chebyshev_tolerance=phi1_auto_chebyshev_tolerance,
                chebyshev_max_degree=phi1_auto_chebyshev_max_degree)
    return output, diagnostics
