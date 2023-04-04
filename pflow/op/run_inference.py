import numpy as np
from typing import List, Dict
from pathlib import Path
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Parameter
)
from matplotlib import pyplot as plt
from pflow.utils import set_directory

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from pflow.nn.transformer import make_transformer

from pflow.nn.flow import make_flow 
from pflow.nn.loss import make_loss
from pflow.utils.files import loaddata
from pflow.common.mol import npz_to_xtc
from pflow.nn import checkpoint
import matplotlib.pyplot as plt 

import sys
import os
import time

class RunInference(OP):

    """`TrainModel` trains a set of neural network models (set by `numb_model` in `train_config`). 
    RiD-kit is powered by TensorFlow framework. The output model files are frozen in `.pb` formats by `rid.nn.freeze`.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "init_data": Artifact(Path),
                "model_log": Artifact(Path),
                "train_config": Dict
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "traj_npz": Artifact(Path),
                "traj_xtc": Artifact(Path)
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        X, traj_length, n, dim, cell, topology = loaddata(op_in["init_data"])
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        
        train_config = op_in["train_config"]
        nheads = train_config["nheads"]
        nlayers = train_config["nlayers"]
        keysize = train_config["keysize"]
        epochs = train_config["epochs"]
        batchsize = train_config["batchsize"]
        lr = train_config["init_lr"]
        frame_dt = train_config["frame_dt"]
        L = cell[0][0][0,0]
        params, vec_field_net = make_transformer(subkey, n, dim, nheads, nlayers, keysize, L)
        sampler = make_flow(vec_field_net, n*dim, L)
        
        print("\n========== Prepare logs ==========")
        folder = os.path.dirname(op_in["model_log"])
        ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(op_in["model_log"])
        print ('folder:', folder)
        if ckpt_filename is not None:
            print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
            ckpt = checkpoint.load_data(ckpt_filename)
            params = ckpt["params"]
        else:
            raise ValueError("no checkpoint found")
        
        task_path = Path("inference")
        task_path.mkdir(exist_ok=True, parents=True)
        
        with set_directory(task_path):
            print("\n========== Start inference ==========")
            '''
            key, key_x0, key_t = jax.random.split(key, 3)
            x1 = X1[:args.batchsize]
            x0 = jax.random.uniform(key_x0, x1.shape, minval=0, maxval=L)
            t = jax.random.uniform(key_t, (args.batchsize,))
            _, loss_fn = make_loss(vec_field_net, L)
            loss = loss_fn(params, x0, x1, t)
            print (t.shape, loss.shape)
            plt.plot(t, loss, 'o')
            plt.show()
            '''
            print("init data shape",X.shape)
            key, subkey = jax.random.split(key)
            print("first atom",X[0][0][:3])
            x = sampler(params, X)
            print ('sample shape', x.shape)
            position = x[0]
            position = position.reshape(position.shape[0],-1,3)
            np.savez("traj_data.npz",positions=position,box = cell,topology=topology)
            npz_to_xtc("traj_data.npz","traj.xtc")
            
        
        op_out = OPIO(
            {
                "traj_npz": task_path.joinpath("traj_data.npz"),
                "traj_xtc": task_path.joinpath("traj.xtc")
            }
        )
        return op_out