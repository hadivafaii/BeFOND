"""Miguel's damped diagonal sum-product GAMP and factor-belief Fisher learning.

This is not an MF ELBO update or a claim of an exact likelihood gradient at
finite depth. The covariance correction and simultaneous moment targets are
part of the method, not optional performance switches.
"""
import sys

from . import shared
import jax
import jax.numpy as jnp
import optax

NAME = 'gamp'


def bernoulli_channel(logits, pseudo, precision):
    r = jax.nn.sigmoid(logits+precision*(pseudo-.5))
    return r, r*(1-r)


def infer(p, x, steps=20, damping=.5):
    w, w2, noise = p['w'], p['w']**2, jnp.exp(p['logvar'])
    pi = jax.nn.sigmoid(p['logits'])

    def output(m, old_s):
        v = m*(1-m)
        vp = v@w2
        pp = m@w-vp*old_s
        ts = 1/(noise+vp)
        sp = (x-p['b']-pp)*ts
        return v, vp, ts, sp

    def body(_, carry):
        m, old_s = carry
        _, _, ts, sp = output(m, old_s)
        s = (1-damping)*old_s+damping*sp
        precision = jnp.maximum(ts@w2.T, 1e-8)
        drive = s@w.T
        # Do not simplify to precision*(m-.5)+drive: different FP rounding.
        mp, _ = bernoulli_channel(p['logits'], m+drive/precision, precision)
        return (1-damping)*m+damping*mp, s

    m = jnp.broadcast_to(pi, (len(x), len(w)))
    s = jnp.zeros_like(x)
    m, s = jax.lax.fori_loop(0, steps, body, (m, s), unroll=1)
    v, vp, ts, sp = output(m, s)
    score = m.T@sp/len(x)+w*(v.T@((sp-s)*sp-ts)/len(x))
    return dict(r=m, score=score, bias_target=p['b']+noise*sp.mean(0),
                noise_target=jnp.mean(noise**2*sp**2+noise*vp*ts, 0),
                disagreement=jnp.mean(jnp.abs(sp-s)))


def initial_state(p, cfg):
    return optax.adam(1.).init(p['w']), jnp.asarray(0, jnp.int32)


def update(p, state, x, cfg, axis_name=None):
    optimizer, count = state
    mean = (lambda a: jax.lax.pmean(a, axis_name)) if axis_name else lambda a: a
    # Keep this FP32 schedule expression consistent across full runs/resumes.
    progress = jnp.clip(count/jnp.maximum(cfg.tokens//cfg.batch-1, 1), 0., 1.)
    factor = .1+.9*.5*(1+jnp.cos(jnp.pi*progress))
    rate, block_rate = cfg.w_lr*factor, cfg.moment_lr*factor
    q = infer(p, x, cfg.steps, cfg.damping)
    grad = -mean(q['score'])
    updates, optimizer = optax.adam(rate).update(grad, optimizer, p['w'])
    w = optax.apply_updates(p['w'], updates)
    w = w*jnp.minimum(1., cfg.norm_max/jnp.maximum(jnp.linalg.norm(w, axis=1), 1e-12))[:, None]
    fraction = -jnp.expm1(-block_rate)
    bt, vt = mean(q['bias_target']), mean(q['noise_target'])
    b = p['b']+fraction*(bt-p['b'])
    var = jnp.exp(p['logvar'])+fraction*(vt-jnp.exp(p['logvar']))
    pi = jax.nn.sigmoid(p['logits'])
    pt = mean(q['r'].mean(0))
    pi = jnp.clip(pi+fraction*(pt-pi), 1e-8, 1-1e-7)
    new = dict(w=w, b=b, logvar=jnp.log(jnp.maximum(var, 1e-6)),
               logits=jnp.log(pi)-jnp.log1p(-pi))
    aux = dict(expected_l0=q['r'].sum(1).mean(), natural_l0=(q['r']>.5).sum(1).mean(),
        prior_l0=jax.nn.sigmoid(new['logits']).sum(), variance=jnp.exp(new['logvar']).mean(),
        reconstruction_mse=jnp.mean((x-new['b']-q['r']@new['w'])**2),
        disagreement=q['disagreement'], max_column_norm=jnp.linalg.norm(new['w'], axis=1).max(),
        score_objective=jnp.asarray(0., x.dtype), parameters_finite=shared.finite(new),
        dictionary_gradient_norm=jnp.linalg.norm(grad), dictionary_rate=rate, moment_rate=block_rate)
    return new, (optimizer, count+1), aux


def probabilities(p, x, cfg):
    return infer(p, x, cfg.steps, cfg.damping)['r']


if __name__ == '__main__':
    shared.main(sys.modules[__name__])
