import os, sys
import logging
from typing import Dict, List
from pathlib import Path
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Parameter,
    BigParameter
)
from pflow.constants import (
    traj_npz_name
    )

from pflow.utils import set_directory, read_txt
import numpy as np
import mdtraj as md

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

class CheckDataInputs(OP):

    r"""Check Inputs of Data Steps.
    
    If inputs `conf` are empty or None, `if_continue` will be False,
    and the following ops of Label steps won't be executed.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "succeeded_task_names":  Artifact(List[Path], optional=True),
                "task_names": BigParameter(List)
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "succeeded_task_names": BigParameter(List),
                "task_names": BigParameter(List)
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
            - `confs`: (`Artifact(List[Path])`) Conformations selected from trajectories of exploration steps.
            
        Returns
        -------
            Output dict with components:
            - `if_continue`: (`bool`) Whether to execute following ops of Label steps.
        """
        succeeded_task_names = []
        for task_name in op_in["succeeded_task_names"]:
            if task_name is not None:
                succeeded_task_names.append(read_txt(task_name))
        task_names = op_in["task_names"]

        op_out = OPIO(
            {
                "succeeded_task_names": succeeded_task_names,
                "task_names":task_names
            }
        )
        return op_out

class PrepData(OP):

    """
    In `PrepData`, labeling processes are achieved by restrained MD simulations 
    where harmonnic restraints are exerted on collective variables.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "conf_begin": Artifact(Path),
                "trajectory_aligned": Artifact(Path),
                "task_name": BigParameter(str)
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "traj_npz": Artifact(Path, archive = None)
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
        task_path = Path(op_in["task_name"])
        task_path.mkdir(exist_ok=True, parents=True)
        with set_directory(task_path):
            # Load the trajectory with mdtraj
            traj = md.load_xtc(op_in["trajectory_aligned"], top=op_in["conf_begin"])

            # Extract the positions and box vectors
            positions = traj.xyz
            box = traj.unitcell_vectors
            
            # Extract topology data from trajectory
            topology = traj.top.to_openmm()

            # Save the positions and box vectors to a npz file
            np.savez(traj_npz_name, positions=positions, box=box, topology = topology)

        op_out = OPIO(
            {
                "traj_npz": task_path.joinpath(traj_npz_name)
            }
        )
        return op_out