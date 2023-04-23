import os, sys
import logging
from typing import Dict, List
from pathlib import Path
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Parameter
)
from pflow.constants import (
    traj_npz_name,
    combined_npz_name
    )

from pflow.utils import set_directory
import numpy as np
import mdtraj as md
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class CombineData(OP):

    """
    In `PrepData`, labeling processes are achieved by restrained MD simulations 
    where harmonnic restraints are exerted on collective variables.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "traj_npz": Artifact(List[Path])
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "combined_npz": Artifact(Path),
                "neighbot_dis_fig": Artifact(Path, archive=None)
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

            - `conf_begin`: (`Artifact(Path)`) Path for the first frames of aligned traj.
            - `trajectory_aligned`: (`Artifact(Path)`) Path of aligned traj.
          
        Returns
        -------
            Output dict with components:
        
            - `traj_npz`: (`Artifact(Path)`) Compressed npz file for aligned traj.
        """
        task_path = Path("combined")
        task_path.mkdir(exist_ok=True, parents=True)
        def squared_distance(traj):
            return jnp.sum((traj[:-1] - traj[1:])**2, axis=-1)
        with set_directory(task_path):
            positions_list = []
            box_list = []
            topology = None

            for npz_file in op_in["traj_npz"]:
                # Load the NPZ file
                if npz_file is not None:
                    data = np.load(npz_file, allow_pickle=True)

                    # Append positions and box dimensions to their respective lists
                    positions_list.append(data['positions'])
                    box_list.append(data['box'])

                    # Use the first file's topology, assuming all files have the same topology
                    if 'topology' in data and topology is None:
                        topology = data['topology']

            # Stack positions and box dimensions along a new axis
            combined_positions = np.stack(positions_list, axis=0)
            combined_box = np.stack(box_list, axis=0)

            # Save the combined data into a new NPZ file
            if topology is not None:
                np.savez(combined_npz_name, positions=combined_positions, box=combined_box, topology=topology)
            else:
                np.savez(combined_npz_name, positions=combined_positions, box=combined_box)
                
            traj_num = combined_positions.shape[0]
            combined_positions = combined_positions.reshape(traj_num,combined_positions.shape[1],\
                combined_positions.shape[2]*combined_positions.shape[3])
            all_dis_list = jax.vmap(squared_distance, in_axes=(0),out_axes = (0))(combined_positions)
            # traj_num, simu_frames
            M,N = all_dis_list.shape
            
            xlist = [i for i in range(N)]
            plt.figure(1)
            for i in range(N):
                plt.scatter([xlist[i]]*M, all_dis_list.T[i])
            plt.xlabel("time (ps)")
            plt.ylabel("neigborig distance (nm)")
            plt.savefig("all_data.png")

        op_out = OPIO(
            {
                "combined_npz": task_path.joinpath(combined_npz_name),
                "neighbot_dis_fig": task_path.joinpath("all_data.png")
            }
        )
        return op_out