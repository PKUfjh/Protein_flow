import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.experimental import ode
from functools import partial

def make_flow(vec_field_net, dim, mxstep=1000):
    
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def forward(params, x0):
        def _ode(x, t):
            result = vec_field_net(params, x, t)
            print("result type",result)
            return result
        xt = ode.odeint(_ode,
                 jnp.asarray(x0,dtype=jnp.float64),
                 jnp.linspace(0, 1, 100), 
                 rtol=1e-10, atol=1e-10,
                 mxstep=mxstep
                 )
        return xt

    return forward
