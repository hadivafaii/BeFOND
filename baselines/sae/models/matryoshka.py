import torch
import torch.nn.functional as F
from torch import Tensor

from baselines.sae.models.base import SAEOutput
from baselines.sae.models.topk import BatchTopKSAE


class MatryoshkaBatchTopKSAE(BatchTopKSAE):
	"""BatchTopK SAE trained with equally weighted nested reconstructions."""

	def __init__(self, config):
		super().__init__(config)
		if config.matryoshka_weights is None:
			weights = torch.full(
				(len(config.matryoshka_prefix_sizes),),
				1.0 / len(config.matryoshka_prefix_sizes),
			)
		else:
			weights = torch.tensor(config.matryoshka_weights)
		self.register_buffer('prefix_weights', weights)

	def forward(
		self,
		x: Tensor,
		dead_mask: Tensor | None = None,
	) -> SAEOutput:
		x_flat = self._flatten_input(x)
		preactivations = self._encode_preactivations(x_flat)
		positive_preactivations = F.relu(preactivations)
		latents = self._sparsify(preactivations)

		prefix_losses = []
		for prefix_size in self.config.matryoshka_prefix_sizes:
			prefix_reconstruction = F.linear(
				latents[:, :prefix_size],
				self.W_dec[:prefix_size].T,
			) + self.b_dec
			prefix_losses.append(self._fraction_variance_unexplained(
				x_flat,
				prefix_reconstruction,
			))
		reconstruction_flat = self._decode_flat(latents)
		matryoshka_loss = (
			torch.stack(prefix_losses) * self.prefix_weights
		).sum()
		loss_terms = {
			'reconstruction': matryoshka_loss,
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


# Concise public name retained for sweep configurations and backwards compatibility.
MatryoshkaSAE = MatryoshkaBatchTopKSAE
