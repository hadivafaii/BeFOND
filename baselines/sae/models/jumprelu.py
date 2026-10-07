import torch
import torch.nn.functional as F
from torch import Tensor, nn

from baselines.sae.models.base import BaseSAE, SAEOutput


def _rectangle(x: Tensor) -> Tensor:
	return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class _JumpReLUFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float):
		ctx.save_for_backward(x, threshold)
		ctx.bandwidth = bandwidth
		return x * (x > threshold).to(x.dtype)

	@staticmethod
	def backward(ctx, grad_output: Tensor):
		x, threshold = ctx.saved_tensors
		bandwidth = ctx.bandwidth
		x_grad = (x > threshold).to(x.dtype) * grad_output
		threshold_grad = (
			-(threshold / bandwidth)
			* _rectangle((x - threshold) / bandwidth)
			* grad_output
		).sum(dim=0)
		return x_grad, threshold_grad, None


class _StepFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float):
		ctx.save_for_backward(x, threshold)
		ctx.bandwidth = bandwidth
		return (x > threshold).to(x.dtype)

	@staticmethod
	def backward(ctx, grad_output: Tensor):
		x, threshold = ctx.saved_tensors
		bandwidth = ctx.bandwidth
		threshold_grad = (
			-(1.0 / bandwidth)
			* _rectangle((x - threshold) / bandwidth)
			* grad_output
		).sum(dim=0)
		return torch.zeros_like(x), threshold_grad, None


class JumpReLUSAE(BaseSAE):
	def __init__(self, config):
		super().__init__(config)
		self.log_threshold = nn.Parameter(torch.full(
			(self.num_latents,),
			torch.log(torch.tensor(config.jump_threshold)).item(),
		))

	@property
	def threshold(self) -> Tensor:
		return self.log_threshold.exp()

	def _jump_activations(
		self,
		x_flat: Tensor,
	) -> tuple[Tensor, Tensor]:
		preactivations = F.relu(self._encode_preactivations(x_flat))
		latents = _JumpReLUFunction.apply(
			preactivations,
			self.threshold,
			self.config.jump_bandwidth,
		)
		return latents, preactivations

	def encode(self, x: Tensor) -> Tensor:
		x_flat = self._flatten_input(x)
		latents, _ = self._jump_activations(x_flat)
		return latents

	def forward(
		self,
		x: Tensor,
		dead_mask: Tensor | None = None,
	) -> SAEOutput:
		del dead_mask
		x_flat = self._flatten_input(x)
		latents, preactivations = self._jump_activations(x_flat)
		reconstruction_flat = self._decode_flat(latents)
		l0 = _StepFunction.apply(
			preactivations,
			self.threshold,
			self.config.jump_bandwidth,
		).sum(dim=1).mean()
		loss_terms = {
			'reconstruction': self._reconstruction_loss(
				x_flat, reconstruction_flat),
			'sparsity': self.config.jump_l0_coefficient * l0,
		}
		return SAEOutput(
			reconstruction=self._restore_input_shape(reconstruction_flat, x),
			latents=latents,
			preactivations=preactivations,
			loss_terms=loss_terms,
		)

