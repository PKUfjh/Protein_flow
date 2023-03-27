from pflow.task.task import Task
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Sequence
import numpy as np
from pflow.constants import (
        gmx_conf_name,
        gmx_top_name,
        gmx_mdp_name, 
        plumed_input_name,
    )
from pflow.utils import read_txt
from pflow.common.gromacs import make_md_mdp_string
from pflow.common.mol import get_distance_from_atomid
from pflow.common.plumed import make_plain_plumed, make_restraint_plumed, get_cv_name


class TaskBuilder(ABC):

    @abstractmethod
    def build(self):
        pass


class CMDTaskBuilder(TaskBuilder):
    def __init__(
        self,
        conf: str,
        topology: Optional[str],
        cmd_config: Dict,
        cv_file: Optional[List[str]] = None,
        selected_resid: Optional[List[int]] = None,
        selected_atomid: Optional[List[int]] = None,
        plumed_output: str = "plm.out",
        cv_mode: str = "torsion"
    ):
        super().__init__()
        self.conf = conf
        self.topology = topology
        self.cmd_config = cmd_config
        self.stride = self.cmd_config["output_freq"]
        self.cv_file = cv_file
        self.selected_resid = selected_resid
        self.selected_atomid = selected_atomid
        self.plumed_output = plumed_output
        self.cv_mode = cv_mode
        self.task = Task()
        self.cv_names = get_cv_name(
            conf=self.conf, cv_file=self.cv_file,
            selected_resid=self.selected_resid,
            selected_atomid=self.selected_atomid,
            stride=self.stride,
            mode=self.cv_mode
        )
    
    def build(self) -> Task:
        task_dict = {}
        task_dict.update(self.build_gmx())
        task_dict.update(self.build_plumed())
        for fname, fconts in task_dict.items():
            self.task.add_file(fname, fconts)
        return self.task
    
    def get_task(self):
        return self.task
     
    def build_gmx(self):
        return build_gmx_dict(self.conf, self.topology, self.cmd_config)
        
    def build_plumed(self):
        return build_plumed_dict(
            conf=self.conf, cv_file=self.cv_file, selected_resid=self.selected_resid,
            selected_atomid=self.selected_atomid,
            stride=self.stride, output=self.plumed_output, mode=self.cv_mode
        )
    
    def get_cv_dim(self):
        return len(self.cv_names)

class RestrainedMDTaskBuilder(TaskBuilder):
    def __init__(
        self,
        conf: str,
        topology: Optional[str],
        label_config: Dict,
        cv_file: Optional[List[str]] = None,
        selected_resid: Optional[List[int]] = None,
        selected_atomid: Optional[List[int]] = None,
        sampler_type: str = "gmx",
        kappa: Union[int, float, Sequence, np.ndarray] = 0.5,
        step: Union[int, float, Sequence, np.ndarray] = 500000,
        nsteps: Union[int, float, Sequence, np.ndarray] = 500000,
        final: Union[int, float, Sequence, np.ndarray] = 10.0,
        plumed_output: str = "plm.out",
        cv_mode: str = "torsion"
    ):
        super().__init__()
        self.conf = conf
        self.topology = topology
        self.label_config = label_config
        self.stride = self.label_config["output_freq"]
        self.cv_file = cv_file
        self.selected_resid = selected_resid
        self.selected_atomid = selected_atomid
        self.plumed_output = plumed_output
        self.cv_mode = cv_mode
        self.sampler_type = sampler_type
        self.kappa = kappa
        self.step = step
        self.nsteps = nsteps
        self.final = final
        self.task = Task()
    
    def build(self) -> Task:
        task_dict = {}
        if self.sampler_type == "gmx":
            task_dict.update(self.build_gmx())
        elif self.sampler_type == "lmp":
            task_dict.update(self.build_lmp())
        task_dict.update(self.build_plumed())
        for fname, fconts in task_dict.items():
            self.task.add_file(fname, fconts)
        return self.task
    
    def get_task(self):
        return self.task
     
    def build_gmx(self):
        return build_gmx_dict(self.conf, self.topology, self.label_config)
        
    def build_plumed(self):
        return build_plumed_restraint_dict(
            conf=self.conf, cv_file=self.cv_file, selected_resid=self.selected_resid,
            selected_atomid=self.selected_atomid, kappa=self.kappa, step=self.step,nsteps=self.nsteps,
            final=self.final,stride=self.stride, output=self.plumed_output, mode=self.cv_mode
        )

def build_gmx_dict(
        conf: str,
        topology: str,
        gmx_config: Dict
    ):
    gmx_task_files = {}
    gmx_task_files[gmx_conf_name] = (read_txt(conf), "w")
    gmx_task_files[gmx_top_name]  = (read_txt(topology), "w")
    mdp_string = make_md_mdp_string(gmx_config)
    gmx_task_files[gmx_mdp_name]  = (mdp_string, "w")
    return gmx_task_files

def build_plumed_dict(
        conf: Optional[str] = None,
        cv_file: Optional[str] = None,
        selected_resid: Optional[List[int]] = None,
        selected_atomid: Optional[List[int]] = None,
        stride: int = 100,
        output: str = "plm.out",
        mode: str = "torsion"
    ):
    plumed_task_files = {}
    plm_content = make_plain_plumed(
        conf=conf, cv_file=cv_file, selected_resid=selected_resid,
        selected_atomid = selected_atomid, stride=stride,
        output=output, mode=mode
    )
    plumed_task_files[plumed_input_name] = (plm_content, "w")
    return plumed_task_files

def build_plumed_restraint_dict(
        conf: Optional[str] = None,
        cv_file: Optional[str] = None,
        selected_resid: Optional[List[int]] = None,
        selected_atomid: Optional[List[int]] = None,
        kappa: Union[int, float, Sequence, np.ndarray] = 0.5,
        step: Union[int, float, Sequence, np.ndarray] = 500000,
        nsteps: Union[int, float, Sequence, np.ndarray] = 500000,
        final: Union[int, float, Sequence, np.ndarray] = 10.0,
        stride: int = 100,
        output: str = "plm.out",
        mode: str = "torsion"
    ):
    plumed_task_files = {}
    if selected_atomid is not None:
        at = []
        cv_info = get_distance_from_atomid(conf, selected_atomid)
        for dis_id in range(len(selected_atomid)):
            at.append(cv_info["%s %s"%(selected_atomid[dis_id][0],selected_atomid[dis_id][1])])
    plm_content = make_restraint_plumed(
        conf=conf, cv_file=cv_file, selected_resid=selected_resid,selected_atomid = selected_atomid,
        kappa=kappa, step=step, nsteps=nsteps,at=at, final=final, stride=stride,
        output=output, mode=mode
    )
    plumed_task_files[plumed_input_name] = (plm_content, "w")
    return plumed_task_files