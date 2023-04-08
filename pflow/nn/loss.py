import jax
import jax.numpy as jnp
from functools import partial
from jax import device_get

def make_loss(vec_field_net, L):

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    def _matching(params, xt, xt_minus_dt, t, dt):
        v = xt_minus_dt - xt
        result = dt*vec_field_net(params, xt, t)
        return jnp.sum((v - result)**2)

    def loss(params, xt, xt_minus_dt, t, dt):
        m = _matching(params, xt, xt_minus_dt, t, dt)
        return jnp.mean(m)

    return loss

def make_mle_loss(logp_fn):
    def loss_fn(params, x, key):
        logp = logp_fn(params, x, key)
        return -jnp.mean(logp)
    return loss_fn
