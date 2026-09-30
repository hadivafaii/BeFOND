"""Ground-truth feature recovery, stratified by fixed empirical frequency."""

import math

import torch

from befond.data.synthetic import _isolated_rng
from .matching import match_dictionary_atoms


FREQUENCY_SAMPLES = 500_000
FREQUENCY_BATCH_SIZE = 1024
FREQUENCY_SEED = 315_031


@torch.no_grad()
def feature_frequency_reference(synthetic_model, samples=FREQUENCY_SAMPLES, seed=FREQUENCY_SEED):
	"""Count actual post-hierarchy firings once, independently of fit/eval seeds.

	Only the frequency vector is cached; no observations or latent codes are
	retained. Sampling restores the caller's random state.
	"""
	cached = getattr(synthetic_model, '_befond_feature_frequencies', {}).get((samples, seed))
	if cached is not None:
		return cached
	vectors = synthetic_model.feature_dict.feature_vectors
	counts = torch.zeros(len(vectors), dtype=torch.int64, device=vectors.device)
	with _isolated_rng(seed, vectors.device):
		for start in range(0, samples, FREQUENCY_BATCH_SIZE):
			batch = synthetic_model.activation_generator.sample(
				min(FREQUENCY_BATCH_SIZE, samples - start))
			if batch.is_sparse:
				batch = batch.to_dense()
			counts += (batch > 0).sum(dim=0)
	frequencies = counts.cpu().double() / samples
	if not hasattr(synthetic_model, '_befond_feature_frequencies'):
		synthetic_model._befond_feature_frequencies = {}
	synthetic_model._befond_feature_frequencies[(samples, seed)] = frequencies
	return frequencies


class GroundTruthFeatureMetrics:
	"""Positive-cosine recovery with missing features scored zero.

	Matching uses dictionary cosine without test labels. Hungarian and Top-k
	are one-to-one; max cosine independently matches each GT feature.
	F1 is macro-averaged over *all* GT features in a group, including unmatched
	and unobserved features. Support counts expose insufficient evaluation data.
	"""

	@torch.no_grad()
	def __init__(self, decoder, truth_vectors, frequencies, matching_method='hungarian'):
		self.frequencies = torch.as_tensor(frequencies).detach().double().cpu()
		n_features = len(truth_vectors)
		if self.frequencies.shape != (n_features,):
			raise ValueError('frequencies must contain one value per GT feature')
		learned, truth, cosines = match_dictionary_atoms(
			decoder, truth_vectors, matching_method)
		positive = cosines > 0
		self.learned_indices = learned[positive]
		self.truth_indices = truth[positive]
		self.cosines = torch.zeros(n_features, dtype=torch.float64)
		self.cosines[self.truth_indices] = cosines[positive]
		self.tp = torch.zeros_like(self.cosines)
		self.predicted = torch.zeros_like(self.cosines)
		self.positives = torch.zeros_like(self.cosines)
		order = torch.argsort(self.frequencies, stable=True)
		self.groups = {
			'gt': order,
			'rare': order[:math.ceil(n_features / 4)],
			**{
				f'frequency_bin_{i}': indices
				for i, indices in enumerate(torch.tensor_split(order, 5), start=1)
			},
		}

	@torch.no_grad()
	def add_batch(self, codes, truth_codes):
		truth_fires = truth_codes.flatten(start_dim=1) > 0
		self.positives += truth_fires.sum(dim=0).cpu()
		fires = codes.flatten(start_dim=1)[:, self.learned_indices.to(codes.device)] > 0
		matched_truth = truth_fires[:, self.truth_indices.to(truth_codes.device)].to(fires.device)
		self.tp[self.truth_indices] += (fires & matched_truth).sum(dim=0).cpu()
		self.predicted[self.truth_indices] += fires.sum(dim=0).cpu()

	def compute(self):
		per_feature = {
			'f1': 2 * self.tp / (self.predicted + self.positives).clamp_min(1),
			'precision': self.tp / self.predicted.clamp_min(1),
			'recall': self.tp / self.positives.clamp_min(1),
			'coverage': (self.cosines >= 0.8).double(),
			'cosine': self.cosines,
		}
		result = {}
		for group, indices in self.groups.items():
			prefix = f'{group}/' if group.startswith('frequency_bin_') else f'{group}_'
			for name, values in per_feature.items():
				result[prefix + name] = float(values[indices].mean()) if len(indices) else 0.0
			frequencies = self.frequencies[indices]
			result.update({
				prefix + 'num_features': len(indices),
				prefix + 'frequency_min': float(frequencies.min()) if len(indices) else 0.0,
				prefix + 'frequency_max': float(frequencies.max()) if len(indices) else 0.0,
				prefix + 'positive_count': int(self.positives[indices].sum()),
				prefix + 'unobserved_features': int((self.positives[indices] == 0).sum()),
			})
		return result
