import torch
import torch.nn.functional as F
from torch import Tensor, nn

from baselines.sae.models.base import BaseSAE, SAEOutput


class GatedSAE(BaseSAE):
	"""Gated SAE with separate feature selection and magnitude paths."""

	def __init__(self, config):
		super().__init__(config)
		self.log_magnitude_scale = nn.Parameter(torch.zeros(self.num_latents))
		self.gate_bias = nn.Parameter(torch.zeros(self.num_latents))
		self.magnitude_bias = nn.Parameter(torch.zeros(self.num_latents))
		# The shared base-class bias is not part of the Gated SAE parameterization.
		self.b_enc.requires_grad_(False)

	def _gated_activations(
		self,
		x_flat: Tensor,
	) -> tuple[Tensor, Tensor, Tensor]:
		linear = F.linear(
			x_flat - self.b_dec,
			self.encoder_directions,
		)
		gate_preactivations = linear + self.gate_bias
		gate = (gate_preactivations > 0).to(linear.dtype)
		magnitude_preactivations = (
			self.log_magnitude_scale.exp() * linear + self.magnitude_bias)
		latents = gate * F.relu(magnitude_preactivations)
		return latents, gate_preactivations, magnitude_preactivations

	def encode(self, x: Tensor) -> Tensor:
		x_flat = self._flatten_input(x)
		latents, _, _ = self._gated_activations(x_flat)
		return latents

	def forward(
		self,
		x: Tensor,
		dead_mask: Tensor | None = None,
	) -> SAEOutput:
		del dead_mask
		x_flat = self._flatten_input(x)
		latents, gate_preactivations, magnitude_preactivations = (
			self._gated_activations(x_flat))
		reconstruction_flat = self._decode_flat(latents)

		gate_activations = F.relu(gate_preactivations)
		auxiliary_reconstruction = F.linear(
			gate_activations,
			self.W_dec.detach().T,
		) + self.b_dec.detach()
		loss_terms = {
			'reconstruction': self._reconstruction_loss(
				x_flat, reconstruction_flat),
			'sparsity': self.config.l1_coefficient
			* gate_activations.abs().sum(dim=1).mean(),
			'auxiliary': self._reconstruction_loss(
				x_flat, auxiliary_reconstruction),
		}
		return SAEOutput(
			reconstruction=self._restore_input_shape(reconstruction_flat, x),
			latents=latents,
			preactivations=magnitude_preactivations,
			loss_terms=loss_terms,
		)

