"""Discrete training horizons with a mean calibrated after truncation."""

import re

import numpy as np
from scipy.optimize import brentq


def parse_horizon_dist(value: str):
	"""Canonicalize exp/power mean specifications while retaining the mean text."""
	if value in ('fixed', 'uniform'):
		return value
	match = re.fullmatch(r'(exp|power)[-_]?(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', value)
	if match and 1.0 <= float(match[2]) < float('inf'):
		return f'{match[1]}-{match[2]}'
	raise ValueError('horizon_dist must be fixed, uniform, exp-<mean>, or power-<mean> '
		'with a finite mean >= 1.')


def horizon_name(t_outer: int, distribution: str):
	distribution = parse_horizon_dist(distribution)
	if distribution == 'fixed':
		return f'h-fixed({t_outer})'
	if distribution == 'uniform':
		return f'h-uniform(1,{t_outer})'
	family, mean = distribution.split('-', 1)
	return f'h-exp(1/{mean})' if family == 'exp' else f'h-power(mu={mean})'


def horizon_probabilities(t_outer: int, distribution: str):
	"""Return probabilities on 1,...,t_outer with the requested truncated mean."""
	if t_outer < 1:
		raise ValueError('t_outer must be positive.')
	distribution = parse_horizon_dist(distribution)
	h = np.arange(t_outer, dtype=np.float64)
	if distribution == 'fixed':
		return (h == t_outer - 1).astype(np.float64)
	if distribution == 'uniform':
		return np.full(t_outer, 1.0 / t_outer)
	family, mean = distribution.split('-', 1)
	mean = float(mean)
	if not 1.0 <= mean <= (t_outer + 1) / 2:
		label = 'Exponential' if family == 'exp' else 'Power-law'
		raise ValueError(
			f'{label} horizon mean must be between 1 and '
			f'(t_outer + 1) / 2 = {(t_outer + 1) / 2:g}.')
	if mean == 1.0:
		return (h == 0).astype(np.float64)
	if mean == (t_outer + 1) / 2:
		return np.full(t_outer, 1.0 / t_outer)

	# exp(-alpha * log(h + 1)) gives power-law weights on integer horizons.
	coordinate = h if family == 'exp' else np.log1p(h)

	def probabilities(rate):
		weights = np.exp(-rate * coordinate)
		return weights / weights.sum()

	if family == 'exp':
		# The untruncated geometric rate brackets the truncated solution.
		upper = np.log1p(1.0 / (mean - 1.0))
	else:
		upper = 1.0
		while h @ probabilities(upper) > mean - 1.0:
			upper *= 2
	rate = brentq(lambda rate: h @ probabilities(rate) - (mean - 1.0),
		0.0, upper, xtol=1e-14)
	return probabilities(rate)
