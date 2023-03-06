import os, sys
import logging
from typing import List, Dict, Union
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
from pflow.constants import (
        gmx_conf_name,
        gmx_top_name,
        gmx_mdp_name, 
        gmx_tpr_name,
        plumed_input_name,
        plumed_output_name,
        gmx_mdrun_log,
        gmx_xtc_name,
        gmx_conf_out,
    )
from pflow.utils import run_command, set_directory, list_to_string
from pflow.common.sampler.command import get_grompp_cmd, get_mdrun_cmd


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class RunCMD(OP):

    """Run (biased or brute-force) MD simulations with files provided by `PrepExplore` OP.
    RiD-kit emploies Gromacs/Lammps as MD engine with PLUMED2 plugin (with or without MPI-implement). Make sure work environment 
    has properly installed Gromacs/Lammps with patched PLUMED2. Also, PLUMED2 need an additional `DeePFE.cpp` patch to use bias potential of RiD.
    See `install` in the home page for details.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "task_path": Artifact(Path),
                "forcefield": Artifact(Path, optional=True),
                "cmd_config": Dict,
                "index_file": Artifact(Path, optional=True)
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "plm_out": Artifact(Path),
                "md_log": Artifact(Path),
                "trajectory": Artifact(Path),
                "conf_out": Artifact(Path)
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

            - `task_path`: (`Artifact(Path)`) A directory path containing files for Gromacs/Lammps MD simulations.
            - `cmd_config`: (`Dict`) Configuration of Gromacs/Lammps simulations in exploration steps.
            - `models`: (`Artifact(List[Path])`) Optional. Neural network model files (`.pb`) used to bias the simulation. 
                Run brute force MD simulations if not provided.
          
        Returns
        -------
            Output dict with components:
        
            - `plm_out`: (`Artifact(Path)`) Outputs of CV values (`plumed.out` by default) from exploration steps.
            - `md_log`: (`Artifact(Path)`) Log files of Gromacs/lammps `mdrun` commands.
            - `trajectory`: (`Artifact(Path)`) Trajectory files (`.xtc`). The output frequency is defined in `cmd_config`.
            - `conf_out`: (`Artifact(Path)`) Final frames of conformations in simulations.
        """
        if op_in["index_file"] is None:
            index = None
        else:
            index = op_in["index_file"].name
        
        if "max_warning" in op_in["cmd_config"]:
            max_warning=op_in["cmd_config"]["max_warning"]
        else:
            max_warning = None
        
        if "nt" in op_in["cmd_config"]:
            nt = op_in["cmd_config"]["nt"]
        else:
            nt = None
        
        if "ntmpi" in op_in["cmd_config"]:
            ntmpi = op_in["cmd_config"]["ntmpi"]
        else:
            ntmpi = None
        
        grompp_cmd = get_grompp_cmd(
            sampler_type=op_in["cmd_config"]["type"],
            mdp = gmx_mdp_name,
            conf = gmx_conf_name,
            topology = gmx_top_name,
            output = gmx_tpr_name,
            index = index,
            max_warning=max_warning
        )
        run_cmd = get_mdrun_cmd(
            sampler_type=op_in["cmd_config"]["type"],
            tpr=gmx_tpr_name,
            plumed=plumed_input_name,
            nt=nt,
            ntmpi=ntmpi
        )

        with set_directory(op_in["task_path"]):
            if op_in["forcefield"] is not None:
                os.symlink(op_in["forcefield"], op_in["forcefield"].name)
            if op_in["index_file"] is not None:
                os.symlink(op_in["index_file"], op_in["index_file"].name)

            if grompp_cmd is not None:
                logger.info(list_to_string(grompp_cmd, " "))
                return_code, out, err = run_command(grompp_cmd)
                assert return_code == 0, err
                logger.info(err)
            if run_cmd is not None:
                logger.info(list_to_string(run_cmd, " "))
                return_code, out, err = run_command(run_cmd)
                assert return_code == 0, err
                logger.info(err)
                        
        if op_in["cmd_config"]["type"] == "gmx":
            mdrun_log = gmx_mdrun_log
            traj_name = gmx_xtc_name
            conf_out = gmx_conf_out
        else:
            raise RuntimeError("Invalid sampler type")
            
        op_out = OPIO(
            {
                "plm_out": op_in["task_path"].joinpath(plumed_output_name),
                "md_log": op_in["task_path"].joinpath(mdrun_log),
                "trajectory": op_in["task_path"].joinpath(traj_name),
                "conf_out": op_in["task_path"].joinpath(conf_out),
            }
        )
        return op_out