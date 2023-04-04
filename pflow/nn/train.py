import jax
import jax.numpy as jnp
import optax
import haiku as hk

from pflow.nn import checkpoint
import os
from typing import NamedTuple
import itertools
from jax import device_get

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

def train(key, value_and_grad, nepoch, batchsize, params, X, lr, path, frame_dt):

    # assert (len(X)%batchsize==0)

    @jax.jit
    def step(key, i, state, X):
        key, subkey = jax.random.split(key)
        xt = []
        xt_minus_dt = []
        t_list = []
        train_rate = 1.0
        for traj_index in range(X.shape[0]):  
            # t = jax.random.uniform(subkey, (int(train_rate*X.shape[1]),))
            init_array = jnp.arange(1, X.shape[1])
            shuffled_array = jax.random.permutation(key, init_array)
            t = shuffled_array/(X.shape[1]-1)
            t_list.append(t)
            dt = jnp.full((t.shape[0],),frame_dt)
            for i in range(t.shape[0]):
                xt.append(X[traj_index,(X.shape[1]*t[i]).astype(int),:])
                xt_minus_dt.append(X[traj_index,(X.shape[1]*(t[i]-dt[i])).astype(int),:])

        xt = jnp.array(xt).reshape(-1,X.shape[2])
        xt_minus_dt = jnp.array(xt_minus_dt).reshape(-1,X.shape[2])
        t_list = jnp.array(t_list).reshape(X.shape[0]*t.shape[0],1)
        dt = jnp.full((X.shape[0]*t.shape[0],1),frame_dt)
        value, grad = value_and_grad(state.params, xt, xt_minus_dt, t_list, dt)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value
    
    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params)
    state = TrainingState(params, init_opt_state)

    log_filename = os.path.join(path, "loss.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")
    itercount = itertools.count()
    for epoch in range(1, nepoch+1):
        key, subkey = jax.random.split(key)

        X = jax.random.permutation(subkey, X)
        # X1 = jax.random.permutation(subkey, X1)

        total_loss = 0.0
        counter = 0 
        for batch_index in range(0, len(X), batchsize):
            key, subkey = jax.random.split(key)
            start = batch_index
            end = min(batch_index+batchsize,len(X))
            state, loss = step(subkey, 
                               next(itercount), 
                               state, 
                               X[start:end]
                               )
            total_loss += loss
            counter += 1

        #print (epoch, total_loss/counter)
        f.write( ("%6d" + "  %.6f" + "\n") % (epoch, total_loss/counter) )

        if epoch % 100 == 0:
            ckpt = {"params": state.params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return state.params