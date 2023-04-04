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
from pflow.utils.path import set_directory
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from pflow.nn.transformer import make_transformer

from pflow.nn.loss import make_loss
from pflow.nn.train import train
from pflow.utils.files import loaddata
import argparse
import time
import os


class TrainModel(OP):

    """`TrainModel` trains a set of neural network models (set by `numb_model` in `train_config`). 
    RiD-kit is powered by TensorFlow framework. The output model files are frozen in `.pb` formats by `rid.nn.freeze`.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "model_tags": str,
                "data": Artifact(Path),
                "train_config": Dict
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "model_log": Artifact(Path)
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:

        r"""Execute the OP.
        
        Parameters
        ----------
        op_in : dict
            Input dict with components:

            - `model_tag`: (`str`) Tags for neural network model files. In formats of `model_{model_tag}.pb`.
            - `angular_mask`: (`List`) Angular mask for periodic collective variables. 1 represents periodic, 0 represents non-periodic.
            - `data`: (`Artifact(Path)`) Data files for training. Prepared by `rid.op.prep_data`.
                `data` has the shape of `[number_conf, 2 * dimension_cv]` and contains the CV values and corresponding mean forces.
            - `train_config`: (`Dict`) Configuration to train neural networks, including training strategy and network structures.
          
        Returns
        -------
            Output dict with components:
        
            - `model`: (`Artifact(Path)`) Neural network models in `.pb` formats.
        """
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        X, traj_length, n, dim, L, topology = loaddata(op_in["data"])

        train_config = op_in["train_config"]
        task_path = Path(op_in["model_tags"])
        task_path.mkdir(exist_ok=True, parents=True)
        
        nheads = train_config["nheads"]
        nlayers = train_config["nlayers"]
        keysize = train_config["keysize"]
        epochs = train_config["epochs"]
        batchsize = train_config["batchsize"]
        lr = train_config["init_lr"]
        frame_dt = train_config["frame_dt"]
        params, vec_field_net = make_transformer(subkey, n, dim, nheads, nlayers, keysize, L)
        modelname = "transformer_l_%d_h_%d_k_%d" % (nlayers, nheads, keysize)
        with set_directory(task_path):
            raveled_params, _ = ravel_pytree(params)
            print("# of params: %d" % raveled_params.size)
            loss = make_loss(vec_field_net, L)
            value_and_grad = jax.value_and_grad(loss)
            print("\n========== Prepare logs ==========")
            path = modelname
            os.makedirs(path, exist_ok=True)
            print("Create directory: %s" % path)
            print("\n========== Train ==========")
            start = time.time()
            params = train(key, value_and_grad, epochs, batchsize, params, X, lr, path, frame_dt)
            end = time.time()
            running_time = end - start
            print("training time: %.5f sec" %running_time)
            
        op_out = OPIO(
            {
                "model_log": task_path.joinpath(path)
            }
        )
        return op_out