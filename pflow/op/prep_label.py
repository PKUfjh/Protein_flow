from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Parameter
)

import json
from typing import List, Dict
from pathlib import Path
from pflow.constants import (
        plumed_output_name
    )
from pflow.task.builder import RestrainedMDTaskBuilder


class CheckLabelInputs(OP):

    r"""Check Inputs of Label Steps.
    
    If inputs `conf` are empty or None, `if_continue` will be False,
    and the following ops of Label steps won't be executed.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "confs": Artifact(List[Path], optional=True),
                "conf_tags": Parameter(List, default=[])
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "if_continue": int,
                "conf_tags": List
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

        if op_in["confs"] is None:
            if_continue = 0
            conf_tags = []
        else:
            if_continue = 1

            tags = {}
            for tag in op_in["conf_tags"]:
                if isinstance(tag, Dict):
                    tags.update(tag)
                elif isinstance(tag, str):
                    tags.update(json.loads(tag))
                else:
                    raise RuntimeError("Unkown Error.")
            
            conf_tags = []
            for conf in op_in["confs"]:
                conf_tags.append(str(tags[conf.name]))

        op_out = OPIO(
            {
                "if_continue": if_continue,
                "conf_tags": conf_tags
            }
        )
        return op_out


class PrepLabel(OP):

    r"""Prepare files for Label steps.
    
    Labels of pflow are mean forces, which are calculated by restrained MD algorithm.
    Restrained MD simulations are performed by Gromacs/Lammps with PLUMED2 plugin, so input files are in Gromacs/Lammps format.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "topology": Artifact(Path, optional=True),
                "conf": Artifact(Path),
                "label_config": Dict,
                "label_cv_config": Dict,
                "task_name": str
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_path": Artifact(Path),
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
        
            - `topology`: (`Artifact(Path)`) Topology files (.top) for Restrained MD simulations.
            - `conf`: (`Artifact(Path)`) Conformation files (.gro, .lmp) for Restrained MD simulations.
            - `label_config`: (`Dict`) Configuration in `Dict` format for Gromacs/Lammps run.
            - `label_cv_config`: (`Dict`) Configuration for CV creation.
            - `task_name`: (`str`) Task name used to make sub-dir for tasks.
           
        Returns
        -------
            Output dict with components:
        
            - `task_path`: (`Artifact(Path)`) A directory containing files for Restrained MD.
        """

        cv_file = None
        selected_resid = None
        selected_atomid = None
        if op_in["label_cv_config"]["mode"] == "torsion":
            selected_resid = op_in["label_cv_config"]["selected_resid"]
        elif op_in["label_cv_config"]["mode"] == "distance":
            selected_atomid = op_in["label_cv_config"]["selected_atomid"]
        elif op_in["label_cv_config"]["mode"] == "custom":
            if "selected_resid" in op_in["label_cv_config"]:
                selected_resid = op_in["label_cv_config"]["selected_resid"]
            elif "selected_atomid" in op_in["label_cv_config"]:
                selected_atomid = op_in["label_cv_config"]["selected_atomid"]
            cv_file = op_in["cv_file"]
        
        #print("what is cv", cv_file)
        if op_in["label_config"]["method"] == "restrained":
            gmx_task_builder = RestrainedMDTaskBuilder(
                conf = op_in["conf"],
                topology = op_in["topology"],
                label_config = op_in["label_config"],
                cv_file = cv_file,
                selected_resid = selected_resid,
                selected_atomid = selected_atomid,
                sampler_type = op_in["label_config"]["type"],
                kappa = op_in["label_config"]["kappas"],
                step = op_in["label_config"]["step"],
                nsteps = op_in["label_config"]["nsteps"],
                final = op_in["label_config"]["final"],
                plumed_output = plumed_output_name,
                cv_mode = op_in["label_cv_config"]["mode"]
            )

        gmx_task = gmx_task_builder.build()
        task_path = Path(op_in["task_name"])
        task_path.mkdir(exist_ok=True, parents=True)
        for fname, fconts in gmx_task.files.items():
            with open(task_path.joinpath(fname), fconts[1]) as ff:
                ff.write(fconts[0])
        op_out = OPIO(
            {
                "task_path": task_path
            }
        )
        return op_out