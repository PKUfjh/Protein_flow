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
                "combined_npz": Artifact(Path)
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
        with set_directory(task_path):
            positions_list = []
            box_list = []
            topology = None

            for npz_file in op_in["traj_npz"]:
                # Load the NPZ file
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

        op_out = OPIO(
            {
                "combined_npz": task_path.joinpath(combined_npz_name)
            }
        )
        return op_out