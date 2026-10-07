"""Small analytic and update checks for the independent Bernoulli baselines."""
from dataclasses import replace

import pytest
jax = pytest.importorskip('jax')
pytest.importorskip('optax')
import jax.numpy as jnp
import numpy as np

from baselines.miguel import mf
from baselines.miguel.config import Config


def test_single_latent_mean_field_is_exact_and_updates_finite():
    p = dict(w=jnp.array([[1., -.5]]), b=jnp.array([.1, .2]),
             logvar=jnp.log(jnp.array([.8, 1.2])), logits=jnp.array([-.7]))
    x = jnp.array([[.3, -.2], [1., .4], [-.2, .7]])
    expected = p['logits'] + (x-p['b']) @ (p['w'] / jnp.exp(p['logvar'])).T
    expected -= .5*jnp.sum(p['w']**2 / jnp.exp(p['logvar']), axis=1)
    np.testing.assert_allclose(mf.infer(p, x, steps=3), expected, atol=2e-7)
    cfg = replace(Config(), method=mf.NAME, width=1, dim=2, batch=3, tokens=30)
    new, _, aux = jax.jit(lambda q,s: mf.update(q,s,x,cfg))(p, mf.initial_state(p,cfg))
    assert all(np.isfinite(np.asarray(v)).all() for v in new.values())
    assert np.all(np.asarray(jnp.exp(new['logvar'])) > 0)
    assert bool(aux['parameters_finite'])
