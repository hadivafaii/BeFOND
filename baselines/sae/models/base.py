from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from baselines.sae.config import SAEConfig


@dataclass
class SAEOutput:
	reconstruction: Tensor
	latents: Tensor
	preactivations: Tensor
	loss_terms: dict[str, Tensor]


class BaseSAE(nn.Module):
	"""Common linear-dictionary interface for all standalone SAEs."""

	def __init__(self, config: SAEConfig):
		super().__init__()
		self.config = config
		self.input_dim = config.input_dim
		self.num_latents = config.num_latents

		generator = torch.Generator(device='cpu')
		generator.manual_seed(config.seed)
		decoder = torch.randn(
			self.num_latents,
			self.input_dim,
			generator=generator,
		)
		decoder = F.normalize(decoder, dim=1)
		self.W_dec = nn.Parameter(decoder)
		self.W_enc = nn.Parameter(decoder.clone())
		self.b_enc = nn.Parameter(torch.zeros(self.num_latents))
		self.b_dec = nn.Parameter(torch.zeros(self.input_dim))

	@property
	def encoder_directions(self) -> Tensor:
		"""Effective encoder rows, shaped ``[num_latents, input_dim]``."""
		if not self.config.aligned:
			return self.W_enc
		decoder_norm_sq = self.W_dec.square().sum(dim=1, keepdim=True)
		decoder_norm_sq = decoder_norm_sq.clamp_min(
			torch.finfo(self.W_dec.dtype).eps)
		dot = (self.W_enc * self.W_dec).sum(dim=1, keepdim=True)
		return self.W_enc + (1.0 - dot) / decoder_norm_sq * self.W_dec

	@property
	def decoder_directions(self) -> Tensor:
		"""Decoder rows, shaped ``[num_latents, input_dim]``."""
		return self.W_dec

	def _flatten_input(self, x: Tensor) -> Tensor:
		if x.ndim == 2 and x.shape[1] == self.input_dim:
			return x
		if (
			x.ndim == len(self.config.input_shape) + 1
			and tuple(x.shape[1:]) == self.config.input_shape
		):
			return x.reshape(x.shape[0], self.input_dim)
		raise ValueError(
			f'Expected [batch, {self.input_dim}] or '
			f'[batch, {self.config.input_shape}], got {tuple(x.shape)}.')

	def _restore_input_shape(self, x_flat: Tensor, reference: Tensor) -> Tensor:
		if reference.ndim == 2:
			return x_flat
		return x_flat.reshape(reference.shape[0], *self.config.input_shape)

	def _encode_preactivations(self, x_flat: Tensor) -> Tensor:
		centered = x_flat - self.b_dec
		return F.linear(centered, self.encoder_directions, self.b_enc)

	def _decode_flat(self, latents: Tensor, add_bias: bool = True) -> Tensor:
		reconstruction = F.linear(latents, self.W_dec.T)
		if add_bias:
			reconstruction = reconstruction + self.b_dec
		return reconstruction

	def decode(self, latents: Tensor, flatten: bool = False) -> Tensor:
		reconstruction = self._decode_flat(latents)
		if flatten:
			return reconstruction
		return reconstruction.reshape(
			latents.shape[0], *self.config.input_shape)

	def encode(self, x: Tensor) -> Tensor:
		raise NotImplementedError

	@staticmethod
	def _reconstruction_loss(x_flat: Tensor, reconstruction_flat: Tensor) -> Tensor:
		return (x_flat - reconstruction_flat).square().sum(dim=1).mean()

	@staticmethod
	def _total_variance(x_flat: Tensor) -> Tensor:
		centered = x_flat - x_flat.mean(dim=0, keepdim=True)
		return centered.square().sum().clamp_min(
			torch.finfo(x_flat.dtype).eps)

	@classmethod
	def _fraction_variance_unexplained(
		cls,
		x_flat: Tensor,
		reconstruction_flat: Tensor,
	) -> Tensor:
		return (
			(x_flat - reconstruction_flat).square().sum()
			/ cls._total_variance(x_flat)
		)

	def compute_loss(
		self,
		x: Tensor,
		output: SAEOutput | None = None,
		dead_mask: Tensor | None = None,
	) -> dict[str, Tensor]:
		if output is None:
			output = self.forward(x, dead_mask=dead_mask)
		terms = dict(output.loss_terms)
		terms['loss'] = torch.stack(tuple(terms.values())).sum()
		return terms

	def prepare_optimizer_step(self):
		"""Project decoder gradients onto the unit-sphere tangent space."""
		if self.W_dec.grad is None:
			return
		with torch.no_grad():
			norm_sq = self.W_dec.square().sum(dim=1, keepdim=True)
			norm_sq = norm_sq.clamp_min(torch.finfo(self.W_dec.dtype).eps)
			parallel_scale = (
				self.W_dec.grad * self.W_dec
			).sum(dim=1, keepdim=True) / norm_sq
			self.W_dec.grad.sub_(parallel_scale * self.W_dec)

	@torch.no_grad()
	def post_optimizer_step(self):
		"""Restore the unit-norm decoder constraint after optimization."""
		self.W_dec.copy_(F.normalize(self.W_dec, dim=1))

	def forward(
		self,
		x: Tensor,
		dead_mask: Tensor | None = None,
	) -> SAEOutput:
		raise NotImplementedError
