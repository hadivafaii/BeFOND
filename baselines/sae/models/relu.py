import torch.nn.functional as F
from torch import Tensor

from baselines.sae.models.base import BaseSAE, SAEOutput


class ReLUSAE(BaseSAE):
	def encode(self, x: Tensor) -> Tensor:
		x_flat = self._flatten_input(x)
		return F.relu(self._encode_preactivations(x_flat))

	def forward(
		self,
		x: Tensor,
		dead_mask: Tensor | None = None,
	) -> SAEOutput:
		del dead_mask
		x_flat = self._flatten_input(x)
		preactivations = self._encode_preactivations(x_flat)
		latents = F.relu(preactivations)
		reconstruction_flat = self._decode_flat(latents)
		loss_terms = {
			'reconstruction': self._reconstruction_loss(
				x_flat, reconstruction_flat),
			'sparsity': self.config.l1_coefficient
			* latents.abs().sum(dim=1).mean(),
		}
		return SAEOutput(
			reconstruction=self._restore_input_shape(reconstruction_flat, x),
			latents=latents,
			preactivations=preactivations,
			loss_terms=loss_terms,
		)
