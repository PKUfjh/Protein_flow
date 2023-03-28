from logging import raiseExceptions
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)

from typing import List, Dict, Union
from pathlib import Path
from pflow.constants import (
        plumed_output_name
    )
from pflow.task.builder import CMDTaskBuilder


class PrepCMD(OP):

    r"""
    Prepare files for exploration tasks.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "topology": Artifact(Path, optional=True),
                "conf": Artifact(Path),
                "cmd_config": Dict,
                "cmd_cv_config": Dict,
                "task_name": str
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_path": Artifact(Path, archive = None)
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
            - `topology`: (`Artifact(Path)`) Topology files (.top) for Gromacs simulations.
            - `conf`: (`Artifact(Path)`) Conformation files (.gro, .lmp) for Gromacs/Lammps simulations.
            - `exploration_config`: (`Dict`) Configuration in `Dict` format for Gromacs/Lammps run.
            - `task_name`: (`str`) Task name used to make sub-dir for tasks.
           
        Returns
        -------
            Output dict with components:
        
            - `task_path`: (`Artifact(Path)`) A directory containing files for pflow exploration.
        """
        selected_resid = None
        selected_atomid = None
        if op_in["cmd_cv_config"]["mode"] == "torsion":
            selected_resid = op_in["cmd_cv_config"]["selected_resid"]
        elif op_in["cmd_cv_config"]["mode"] == "distance":
            selected_atomid = op_in["cmd_cv_config"]["selected_atomid"]
        
        gmx_task_builder = CMDTaskBuilder(
            conf = op_in["conf"],
            topology = op_in["topology"],
            cmd_config = op_in["cmd_config"],
            selected_resid = selected_resid,
            selected_atomid = selected_atomid,
            plumed_output = plumed_output_name,
            cv_mode = op_in["cmd_cv_config"]["mode"]
        )
        task_path = Path(op_in["task_name"])
        task_path.mkdir(exist_ok=True, parents=True)
        gmx_task = gmx_task_builder.build()
        for fname, fconts in gmx_task.files.items():
            with open(task_path.joinpath(fname), fconts[1]) as ff:
                ff.write(fconts[0])
        op_out = OPIO(
            {
                "task_path": task_path
            }
        )
        return op_out