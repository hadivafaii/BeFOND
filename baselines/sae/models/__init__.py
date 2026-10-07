from baselines.sae.config import SAEConfig
from baselines.sae.models.base import BaseSAE, SAEOutput
from baselines.sae.models.gated import GatedSAE
from baselines.sae.models.jumprelu import JumpReLUSAE
from baselines.sae.models.matryoshka import (
	MatryoshkaBatchTopKSAE,
	MatryoshkaSAE,
)
from baselines.sae.models.relu import ReLUSAE
from baselines.sae.models.topk import BatchTopKSAE, TopKSAE


_MODEL_CLASSES = {
	'relu': ReLUSAE,
	'gated': GatedSAE,
	'topk': TopKSAE,
	'batch_topk': BatchTopKSAE,
	'jumprelu': JumpReLUSAE,
	'matryoshka': MatryoshkaSAE,
}


def build_sae(config: SAEConfig) -> BaseSAE:
	"""Construct the model selected by ``config.model_type``."""
	if not isinstance(config, SAEConfig):
		raise TypeError('config must be an SAEConfig.')
	return _MODEL_CLASSES[config.model_type](config)


__all__ = [
	'BaseSAE',
	'BatchTopKSAE',
	'GatedSAE',
	'JumpReLUSAE',
	'MatryoshkaBatchTopKSAE',
	'MatryoshkaSAE',
	'ReLUSAE',
	'SAEConfig',
	'SAEOutput',
	'TopKSAE',
	'build_sae',
]
