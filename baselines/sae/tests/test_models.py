"""Core contracts shared by the six compact comparison models."""
import pytest
import torch

from baselines.sae import SAEConfig, build_sae
from baselines.sae.checkpoints import load_sae, save_sae


@pytest.mark.parametrize('family', ['relu', 'gated', 'topk', 'batch_topk', 'jumprelu', 'matryoshka'])
def test_training_and_checkpoint_roundtrip(family, tmp_path):
    model = build_sae(SAEConfig(model_type=family, input_shape=(8,), num_latents=16, k_active=3))
    x = torch.randn(12, 8)
    out = model(x, dead_mask=torch.zeros(16, dtype=torch.bool))
    loss = model.compute_loss(x, out)['loss']
    assert torch.isfinite(loss)
    loss.backward()
    model.prepare_optimizer_step()
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    model.post_optimizer_step()
    torch.testing.assert_close(model.W_dec.norm(dim=1), torch.ones(16))
    model.eval()
    expected = model.encode(x)
    save_sae(model, tmp_path)
    actual = load_sae(tmp_path)
    torch.testing.assert_close(actual.encode(x), expected, rtol=0, atol=0)
    assert actual.decode(expected, flatten=True).shape == x.shape


def test_keyed_training_resume_preserves_optimizer_and_schedule(tmp_path):
    from befond.data.toy import ToySource
    from baselines.sae.training import TrainingConfig, train

    source_config = dict(model=dict(input_dim=8, num_latents=16), data=dict(seed=4))
    source = ToySource(source_config)
    cfg = SAEConfig(input_shape=(8,), num_latents=16, k_active=3)
    training = TrainingConfig(train_steps=4, batch_size=12, warmup_steps=1,
                              checkpoint_freq=2, log_freq=4)
    expected = train(build_sae(cfg), source, training, output=tmp_path/'complete')

    class Interrupted:
        def sample(self, batch_size, step):
            if step == 2:
                raise InterruptedError
            return source.sample(batch_size, step)

    with pytest.raises(InterruptedError):
        train(build_sae(cfg), Interrupted(), training, output=tmp_path/'resumed')
    resumed = train(build_sae(cfg), source, training, output=tmp_path/'resumed', resume=True)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
