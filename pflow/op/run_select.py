from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Parameter,
    BigParameter
)

from typing import List, Optional, Union, Dict
from pathlib import Path
from pflow.select.cluster import Cluster
from pflow.utils import save_txt, set_directory
from pflow.common.mol import slice_xtc
from pflow.constants import (
    sel_ndx_name,
    sel_gro_name
)
import numpy as np
import json


class RunSelect(OP):

    """RunSelect OP clusters CV outputs of each parallel walker from exploration steps and prepares representative 
    frames of each clusters for further selection steps.
    RiD-kit employs agglomerative clustering algorithm performed by Scikit-Learn python package. The distance matrix of CVs
    is pre-calculated, which is defined by Euclidean distance in CV space. For each cluster, one representive frame will 
    be randomly chosen from cluster members.
    For periodic collective variables, RiD-kit uses `angular_mask` to identify them and handle their periodic conditions 
    during distance calculation.
    In the first run of RiD iterations, PrepSelect will make a cluster threshold automatically from the initial guess of this value 
    and make cluter numbers of each parallel walker fall into the interval of `[numb_cluster_lower, numb_cluster_upper]`.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "task_name": str,
                "xtc_traj": Artifact(Path),
                "topology": Artifact(Path),
                "plm_out": Artifact(Path),
                "cluster_threshold": float,
                "angular_mask": Optional[Union[np.ndarray, List]],
                "weights": Optional[Union[np.ndarray, List]],
                "numb_cluster_upper": Parameter(Optional[float], default=None),
                "numb_cluster_lower": Parameter(Optional[float], default=None),
                "dt": Parameter(Optional[float], default=None),
                "output_freq": Parameter(Optional[float], default=None),
                "slice_mode": str, 
                "max_selection": int
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "numb_cluster": int,
                "selected_confs": Artifact(List[Path], archive = None),
                "selected_indices": Artifact(Path, archive = None),
                "selected_conf_tags": Artifact(Path, archive= None)
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

            - `task_name`: (`str`) Task names, used to make sub-directory for tasks.
            - `plm_out`: (`Artifact(Path)`) Outputs of CV values (`plumed.out` by default) from exploration steps.
            - `cluster_threshold`: (`float`) Cluster threshold of agglomerative clustering algorithm
            - `angular_mask`: (`array_like`) Mask for periodic collective variables. 1 represents periodic, 0 represents non-periodic.
            - `weights`: (`array_like`) Weights to cluster collective variables. see details in cluster parts.
            - `numb_cluster_upper`: (`Optional[float]`) Upper limit of cluster number to make cluster threshold.
            - `numb_cluster_lower`: (`Optional[float]`) Lower limit of cluster number to make cluster threshold.
            - `max_selection`: (`int`) Max selection number of clusters in Selection steps for each parallel walker.
                For each cluster, one representive frame will be randomly chosen from cluster members.
            - `if_make_threshold`: (`bool`) whether to make threshold to fit the cluster number interval. Usually `True` in the 1st 
                iteration and `False` in the further iterations. 

        Returns
        -------
            Output dict with components:
        
            - `numb_cluster`: (`int`) Number of clusters.
            - `cluster_threshold`: (`float`) Cluster threshold of agglomerative clustering algorithm. 
            - `cluster_selection_index`: (`Artifact(Path)`) Indice of chosen representive frames of clusters in trajectories.
            - `cluster_selection_data`: (`Artifact(Path)`) Collective variable values of chosen representive frames of clusters.
        """

        # the first column of plm_out is time index
        data = np.loadtxt(op_in["plm_out"])[:,1:]
        cv_cluster = Cluster(
            data, op_in["cluster_threshold"], angular_mask=op_in["angular_mask"], 
            weights=op_in["weights"], max_selection=op_in["max_selection"])
        
        threshold = cv_cluster.make_threshold(op_in["numb_cluster_lower"], op_in["numb_cluster_upper"])

        cls_sel_idx = cv_cluster.get_cluster_selection()
        numb_cluster = len(cls_sel_idx)

        walker_idx = int(op_in["task_name"])
        task_path = Path(op_in["task_name"])
        task_path.mkdir(exist_ok=True, parents=True)
        with set_directory(task_path):
            save_txt(sel_ndx_name, cls_sel_idx, fmt="%d")
            if op_in["slice_mode"] == "gmx":
                assert op_in["dt"] is not None, "Please provide time step to slice trajectory."
                for ii, sel in enumerate(cls_sel_idx):
                    time = sel * op_in["dt"] * op_in["output_freq"]
                    slice_xtc(xtc=op_in["xtc_traj"], top=op_in["topology"],
                            walker_idx=walker_idx,selected_idx=time, output=sel_gro_name.format(walker=walker_idx,idx=sel), style="gmx")
            conf_list = []
            conf_tags = {}
            for ii, sel in enumerate(cls_sel_idx):
                if op_in["slice_mode"] == "gmx" or op_in["slice_mode"] == "mdtraj" :
                    conf_list.append(task_path.joinpath(sel_gro_name.format(walker=walker_idx,idx=sel)))
                    conf_tags[sel_gro_name.format(walker = walker_idx,idx=sel)] = f"{op_in['task_name']}_{sel}"
            with open("conf.json", "w") as f:
                json.dump(conf_tags,f)

        
        op_out = OPIO({
                "numb_cluster": numb_cluster,
                "selected_confs": conf_list,
                "selected_indices": task_path.joinpath(sel_ndx_name),
                "selected_conf_tags": task_path.joinpath("conf.json")
            })
        return op_out

    