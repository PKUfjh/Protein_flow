import os, sys, shutil, logging
from typing import List, Dict
from pathlib import Path
from copy import deepcopy
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
from pflow.utils import load_json
from pflow.constants import init_conf_name, walker_tag_fmt


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def prep_confs(confs, numb_walkers):
    numb_confs = len(confs)
    conf_list = []
    if numb_confs < numb_walkers:
        logger.info("Number of confs is smaller than number of walkers. Copy replicas up to number of walkers.")
        for idx in range(numb_walkers):
            shutil.copyfile(confs[idx%numb_confs], init_conf_name.format(idx=idx))
    elif numb_confs > numb_walkers:
        logger.info("Number of confs is greater than number of walkers. Only use the fist `numb_walkers` confs.")
        for idx in range(numb_walkers):
            shutil.copyfile(confs[idx], init_conf_name.format(idx=idx))
    else:
        for idx in range(numb_walkers):
            shutil.copyfile(confs[idx], init_conf_name.format(idx=idx))
    for idx in range(numb_walkers):
        conf_list.append(Path(init_conf_name.format(idx=idx)))
    return conf_list


class PrepPflow(OP):

    """Pre-processing of pflow.
    
    1. Parse pflow configuration JSON file, get default value if parameters are not provided.
    2. Rearrange conformation files.
    3. Make task names and formats.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "confs": Artifact(List[Path]),
                "pflow_config": Artifact(Path)
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "numb_walkers": int,
                "confs": Artifact(List[Path],archive = None),
                "walker_tags": List,
                
                "cmd_config": Dict,
                "cmd_cv_config": Dict,
                "label_cv_config": Dict,
                "cluster_threshold": List[float],
                "angular_mask": List,
                "weights": List,
                "numb_cluster_upper": int,
                "numb_cluster_lower": int,
                "max_selection": int,
                "dt": float,
                "output_freq": float,
                "slice_mode": str,
                "label_config": Dict,
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
        
            - `confs`: (`Artifact(Path)`) User-provided initial conformation files (.gro) for reinfoced dynamics.
            - `pflow_config`: (`Artifact(Path)`) Configuration file (.json) of pflow. 
                Parameters in this file will be parsed.
           
        Returns
        -------
            Output dict with components:
        
            - `numb_iters`: (`int`) Max number of iterations for Ri.
            - `numb_walkers`: (`int`) Number of parallel walkers for exploration.
            - `confs`: (`Artifact(List[Path])`) Rearranged initial conformation files (.gro) for reinfoced dynamics.
            - `walker_tags`: (`List`) Tag formmat for parallel walkers.
            
            - `cmd_config`: (`Dict`) Configuration of simulations in exploration steps.
            - `cv_config`: (`Dict`) Configuration to create CV in PLUMED2 style.
            - `cluster_threshold`: (`List[float]`) Initial guess of cluster threshold.
            - `angular_mask`: (`List`) Angular mask for periodic collective variables. 
                1 represents periodic, 0 represents non-periodic.
            - `weights`: (`List`) Weights for clustering collective variables. see details in cluster algorithms.
            - `numb_cluster_upper`: (`int`) Upper limit of cluster number to make cluster threshold.
            - `numb_cluster_lower`: (`int`) Lower limit of cluster number to make cluster threshold.
            - `dt`: (`float`) Time interval of exploration MD simulations. Gromacs `trjconv` commands will need this parameters 
                to slice trajectories by `-dump` tag, see `selection` steps for detail.
            - `slice_mode`: (`str`) Mode to slice trajectories. Either `gmx` or `mdtraj`.
            - `label_config`: (`Dict`) Configuration of simulations in labeling steps.
        """

        jdata = deepcopy(load_json(op_in["pflow_config"]))
        numb_walkers = jdata.pop("numb_walkers")
        conf_list = prep_confs(op_in["confs"], numb_walkers)

        walker_tags = []
        for idx in range(numb_walkers):
            walker_tags.append(walker_tag_fmt.format(idx=idx))
        
        cmd_config = jdata.pop("cmd_config")
        dt = cmd_config["dt"]
        output_freq = cmd_config["output_freq"]
        cmd_cv_config = jdata.pop("cmd_cv")
        label_cv_config = jdata.pop("label_cv")
        angular_mask = cmd_cv_config["angular_mask"]
        weights = cmd_cv_config["weights"]
        
        selection_config = jdata.pop("select_config")

        label_config = jdata.pop("label_config")
        
        cluster_threshold = selection_config.pop("cluster_threshold")
        cluster_threshold_list = [cluster_threshold for _ in range(numb_walkers)]
        
        op_out = OPIO(
            {
                "numb_walkers": numb_walkers,
                "confs": conf_list,
                "walker_tags": walker_tags,

                "cmd_config": cmd_config,
                "cmd_cv_config": cmd_cv_config,
                "label_cv_config": label_cv_config,
                "cluster_threshold": cluster_threshold_list,
                "angular_mask": angular_mask,
                "weights": weights,
                "numb_cluster_upper": selection_config.pop("numb_cluster_upper"),
                "numb_cluster_lower": selection_config.pop("numb_cluster_lower"),
                "max_selection": selection_config.pop("max_selection"),
                "dt": dt,
                "output_freq": output_freq,
                "slice_mode": selection_config.pop("slice_mode"),
                "label_config": label_config,
            }
        )
        return op_out