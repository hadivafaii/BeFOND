import torch
import torch.nn.functional as F
from torch import Tensor

from baselines.sae.models.base import BaseSAE, SAEOutput


def _scatter_topk(values: Tensor, k: int) -> Tensor:
	selected = values.topk(k, dim=-1, sorted=False)
	return torch.zeros_like(values).scatter(
		dim=-1,
		index=selected.indices,
		src=selected.values,
	)


class TopKSAE(BaseSAE):
	def _sparsify(self, preactivations: Tensor) -> Tensor:
		return _scatter_topk(F.relu(preactivations), self.config.k_active)

	def encode(self, x: Tensor) -> Tensor:
		x_flat = self._flatten_input(x)
		return self._sparsify(self._encode_preactivations(x_flat))

	def _auxiliary_loss(
		self,
		x_flat: Tensor,
		reconstruction_flat: Tensor,
		positive_preactivations: Tensor,
		dead_mask: Tensor | None,
	) -> Tensor:
		zero = x_flat.new_zeros(())
		if self.config.auxk_coefficient == 0 or dead_mask is None:
			return zero
		dead_mask = dead_mask.to(
			device=x_flat.device,
			dtype=torch.bool,
		)
		if dead_mask.shape != (self.num_latents,):
			raise ValueError(
				f'dead_mask must have shape ({self.num_latents},).')
		if not bool(dead_mask.any()):
			return zero

		num_dead = int(dead_mask.sum().item())
		target_k = self.config.auxk_k or self.input_dim // 2
		target_k = max(1, target_k)
		k_aux = min(target_k, num_dead)
		scale = min(num_dead / target_k, 1.0)
		candidates = torch.where(
			dead_mask.unsqueeze(0),
			positive_preactivations,
			-torch.inf,
		)
		aux_latents = _scatter_topk(candidates, k_aux)
		auxiliary_reconstruction = self._decode_flat(
			aux_latents,
			add_bias=False,
		)
		residual = (x_flat - reconstruction_flat).detach()
		numerator = (residual - auxiliary_reconstruction).square().sum()
		return (
			self.config.auxk_coefficient
			* scale
			* numerator
			/ self._total_variance(x_flat)
		)

	def forward(
		self,
		x: Tensor,
		dead_mask: Tensor | None = None,
	) -> SAEOutput:
		x_flat = self._flatten_input(x)
		preactivations = self._encode_preactivations(x_flat)
		positive_preactivations = F.relu(preactivations)
		latents = self._sparsify(preactivations)
		reconstruction_flat = self._decode_flat(latents)
		loss_terms = {
			'reconstruction': self._fraction_variance_unexplained(
				x_flat, reconstruction_flat),
			'auxiliary': self._auxiliary_loss(
				x_flat,
				reconstruction_flat,
				positive_preactivations,
				dead_mask,
			),
		}
		return SAEOutput(
			reconstruction=self._restore_input_shape(reconstruction_flat, x),
			latents=latents,
			preactivations=positive_preactivations,
			loss_terms=loss_terms,
		)


class BatchTopKSAE(TopKSAE):
	def __init__(self, config):
		super().__init__(config)
		self.register_buffer('inference_threshold', torch.tensor(-1.0))

	@torch.no_grad()
	def _update_inference_threshold(self, latents: Tensor):
		positive = latents[latents > 0]
		if positive.numel() == 0:
			return
		threshold = positive.min().to(
			device=self.inference_threshold.device,
			dtype=self.inference_threshold.dtype,
		)
		if self.inference_threshold < 0:
			self.inference_threshold.copy_(threshold)
		else:
			decay = self.config.threshold_decay
			self.inference_threshold.mul_(decay).add_(
				threshold * (1.0 - decay))

	def _sparsify(self, preactivations: Tensor) -> Tensor:
		positive = F.relu(preactivations)
		if not self.training and self.inference_threshold >= 0:
			return positive * (positive >= self.inference_threshold)
		num_selected = min(
			positive.numel(),
			positive.shape[0] * self.config.k_active,
		)
		flat = positive.flatten()
		selected = flat.topk(num_selected, sorted=False)
		latents = torch.zeros_like(flat).scatter(
			0, selected.indices, selected.values).reshape_as(positive)
		if self.training:
			self._update_inference_threshold(latents)
		return latents
