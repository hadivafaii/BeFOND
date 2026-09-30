"""Bernoulli/Gaussian mean-field inference and fixed-q ELBO learning.

Parallel damped MF uses the learned prior throughout inference. Dictionary
learning uses Adam; bias, diagonal variance and prior use moment updates.
"""
import sys

from . import shared
import jax
import jax.numpy as jnp
import optax

from .shared import finite, project

NAME = 'mf'


def fields(p, x):
    weighted = p['w']*jnp.exp(-p['logvar'])
    norm = jnp.sum(p['w']*weighted, axis=1)
    h = p['logits']+(x-p['b'])@weighted.T-.5*norm
    return h, weighted, norm


def proposal(p, f, u):
    h, weighted, norm = f
    r = jax.nn.sigmoid(u)
    return h-((r@p['w'])@weighted.T-r*norm)


def infer(p, x, steps=20, damping=.2, first_damping=1.):
    """Parallel logit-space MF; the first interaction remains batch-shaped."""
    f = fields(p, x)
    prior = jnp.broadcast_to(p['logits'], f[0].shape)
    first = prior+first_damping*(proposal(p, f, prior)-prior)

    def body(u, unused):
        return u+damping*(proposal(p, f, u)-u), None

    return jax.lax.scan(body, first, None, length=steps-1, unroll=True)[0]


def moment_blocks(p, x, u, w, eta, norm_max=4.):
    """Fixed q, sequential W -> bias -> variance -> prior -> norm projection.

    Bias/variance use the unprojected new W.
    Moving the projection earlier would change the learning algorithm.
    """
    r = jax.nn.sigmoid(u)
    pi = jax.nn.sigmoid(p['logits'])
    bt = jnp.mean(x-r@w, axis=0)
    b = p['b']+(-jnp.expm1(-eta))*(bt-p['b'])
    vt = jnp.mean((x-b-r@w)**2+(r*(1-r))@(w*w), axis=0)
    variance = jnp.exp(p['logvar'])
    variance += (-jnp.expm1(-eta))*(vt-variance)
    pt = r.mean(0)
    pi_new = pi+(-jnp.expm1(-eta*1.))*(pt-pi)
    pi_new = jnp.clip(pi_new, 1e-12, 1-1e-7)
    new = project(dict(w=w, b=b, logvar=jnp.log(variance),
                       logits=jnp.log(pi_new)-jnp.log1p(-pi_new)), 0., norm_max)
    return new


def negative_elbo(p, x, u):
    r = jax.nn.sigmoid(u)
    residual = x-p['b']-r@p['w']
    precision = jnp.exp(-p['logvar'])
    norm = jnp.sum(p['w']**2*precision, axis=1)
    error = jnp.sum(residual**2*precision, axis=1)+(r*(1-r))@norm
    logdet = jnp.sum(jnp.broadcast_to(p['logvar'], (x.shape[1],)))
    kl = jnp.sum(r*(u-p['logits'])-jax.nn.softplus(u)+jax.nn.softplus(p['logits']), axis=1)
    return jnp.mean(.5*(error+x.shape[1]*jnp.log(2*jnp.pi)+logdet)+kl)


def schedule(count, cfg):
    horizon = cfg.tokens//cfg.batch
    return .1+.9*.5*(1+jnp.cos(jnp.pi*jnp.minimum(count/max(1, horizon-1), 1.)))


def initial_state(p, cfg):
    opt = optax.adam(1.).init(p['w'])
    if cfg.clip_ratio:
        # Running clipped-gradient norm, running objective and recent skip rate.
        return (opt, jnp.asarray(0, jnp.int32), jnp.asarray(0., jnp.float32),
                jnp.asarray(0., jnp.float32), jnp.asarray(0., jnp.float32))
    return opt, jnp.asarray(0, jnp.int32)


def update(p, state, x, cfg):
    if cfg.clip_ratio:
        return guarded_update(p, state, x, cfg)
    opt, count = state
    factor = schedule(count, cfg)
    rate, eta = cfg.w_lr*factor, cfg.moment_lr*factor
    u = jax.lax.stop_gradient(infer(p, x, cfg.steps, cfg.damping, cfg.first_damping))
    r = jax.nn.sigmoid(u)
    residual = r@p['w']+p['b']-x
    grad = (r.T@residual/len(x)+(r*(1-r)).mean(0)[:, None]*p['w'])*jnp.exp(-p['logvar'])
    updates, opt = optax.adam(1.).update(grad, opt, p['w'])
    w = p['w']+rate*updates
    new = moment_blocks(p, x, u, w, eta, cfg.norm_max)
    # These outputs remain live across the synthetic compiled chunk boundary.
    # Removing them changes XLA fusion and can change long-run FP32 trajectories.
    before = negative_elbo(p, x, u)
    norms = jnp.linalg.norm(new['w'], axis=1)
    variance = jnp.exp(p['logvar'])
    vt = jnp.mean((x-new['b']-r@w)**2+(r*(1-r))@(w*w), axis=0)
    variance += (-jnp.expm1(-eta))*(vt-variance)
    aux = dict(score_objective=before, frozen_nelbo_change=negative_elbo(new, x, u)-before,
        expected_l0=r.sum(1).mean(), natural_l0=(r>.5).sum(1).mean(),
        prior_l0=jax.nn.sigmoid(new['logits']).sum(), variance=variance.mean(),
        variance_min=variance.min(), variance_max=variance.max(),
        mean_column_norm=norms.mean(), min_column_norm=norms.min(), max_column_norm=norms.max(),
        parameters_finite=finite(new), optimizer_finite=finite(opt), wb_rate=rate, moment_rate=eta,
        dictionary_update_norm=jnp.linalg.norm(new['w']-p['w']))
    return new, (opt, count+1), aux


def guarded_update(p, state, x, cfg):
    """update() with dictionary-gradient clipping and skipping of divergent batches.

    From zero-based update guard_warmup on, the gradient norm is clipped at
    clip_ratio times a running mean (EMA .99) of clipped norms, and a batch whose
    negative ELBO exceeds skip_ratio times its running mean (EMA .99 over accepted
    batches) leaves all parameters and optimizer moments unchanged, unless the
    recent skip rate (EMA .99) has reached max_skip_rate. Both running means start
    at their first value. The update count, and hence the schedule, always advances.
    """
    opt, count, ema, objective, skip_rate = state
    opt_before = opt
    factor = schedule(count, cfg)
    rate, eta = cfg.w_lr*factor, cfg.moment_lr*factor
    u = jax.lax.stop_gradient(infer(p, x, cfg.steps, cfg.damping, cfg.first_damping))
    r = jax.nn.sigmoid(u)
    residual = r@p['w']+p['b']-x
    grad = (r.T@residual/len(x)+(r*(1-r)).mean(0)[:, None]*p['w'])*jnp.exp(-p['logvar'])
    gnorm = jnp.linalg.norm(grad)
    scale = jnp.minimum(1., jnp.where(ema > 0, cfg.clip_ratio*ema, gnorm)/gnorm)
    scale = jnp.where(count < cfg.guard_warmup, 1., scale)
    grad = grad*scale
    ema = jnp.where(ema > 0, .99*ema+.01*gnorm*scale, gnorm)
    updates, opt = optax.adam(1.).update(grad, opt, p['w'])
    w = p['w']+rate*updates
    new = moment_blocks(p, x, u, w, eta, cfg.norm_max)
    before = negative_elbo(p, x, u)
    norms = jnp.linalg.norm(new['w'], axis=1)
    variance = jnp.exp(p['logvar'])
    vt = jnp.mean((x-new['b']-r@w)**2+(r*(1-r))@(w*w), axis=0)
    variance += (-jnp.expm1(-eta))*(vt-variance)
    aux = dict(score_objective=before, frozen_nelbo_change=negative_elbo(new, x, u)-before,
        expected_l0=r.sum(1).mean(), natural_l0=(r>.5).sum(1).mean(),
        prior_l0=jax.nn.sigmoid(new['logits']).sum(), variance=variance.mean(),
        variance_min=variance.min(), variance_max=variance.max(),
        mean_column_norm=norms.mean(), min_column_norm=norms.min(), max_column_norm=norms.max(),
        parameters_finite=finite(new), optimizer_finite=finite(opt), wb_rate=rate, moment_rate=eta,
        dictionary_update_norm=jnp.linalg.norm(new['w']-p['w']), grad_norm=gnorm, grad_scale=scale)
    skip = (objective > 0) & (before > cfg.skip_ratio*objective) & (count >= cfg.guard_warmup) \
        & (skip_rate < cfg.max_skip_rate)
    new = jax.tree.map(lambda old, nu: jnp.where(skip, old, nu), p, new)
    opt = jax.tree.map(lambda old, nu: jnp.where(skip, old, nu), opt_before, opt)
    objective = jnp.where(objective > 0, jnp.where(skip, objective, .99*objective+.01*before), before)
    skip_rate = .99*skip_rate+.01*skip.astype(jnp.float32)
    return new, (opt, count+1, ema, objective, skip_rate), dict(aux, skipped=skip)


def probabilities(p, x, cfg):
    return jax.nn.sigmoid(infer(p, x, cfg.steps, cfg.damping, cfg.first_damping))



if __name__ == '__main__':
    shared.main(sys.modules[__name__])
