from distutils.command.config import dump_file
import json
from pathlib import Path
from typing import List, Union, Optional
from pflow.utils import load_json
from copy import deepcopy
import os

from dflow import (
    Workflow,
    Step,
    upload_artifact
)

from dflow.python import upload_packages
from pflow import SRC_ROOT
upload_packages.append(SRC_ROOT)

from pflow.utils import normalize_resources
from pflow.superop.protein_cmd import CMD
from pflow.superop.selector import Selector
from pflow.superop.protein_label import Label
from pflow.superop.protein_data import Data
from pflow.superop.protein_train import Train
from pflow.op.prep_cmd import PrepCMD
from pflow.op.run_cmd import RunCMD
from pflow.op.prep_label import CheckLabelInputs,PrepLabel
from pflow.op.prep_data import CheckDataInputs,PrepData
from pflow.op.combine_data import CombineData
from pflow.op.run_label import RunLabel
from pflow.op.run_select import RunSelect
from pflow.op.run_train import TrainModel
from pflow.flow.loop import ProteinFlow


def prep_pflow_op(
    prep_cmd_config,
    run_cmd_config,
    prep_label_config,
    run_label_config,
    run_select_config,
    prep_data_config,
    combine_data_config,
    train_config,
    workflow_steps_config,
    retry_times
    ):

    cmd_op = CMD(
        "cmd",
        PrepCMD,
        RunCMD,
        prep_cmd_config,
        run_cmd_config,
        retry_times=retry_times)
    
    select_op = Selector(
        "select",
        RunSelect,
        run_select_config,
        retry_times=retry_times)
    
    label_op = Label(
        "label",
        CheckLabelInputs,
        PrepLabel,
        RunLabel,
        prep_label_config,
        run_label_config,
        retry_times=retry_times)

    data_op = Data(
        "data",
        CheckDataInputs,
        PrepData,
        CombineData,
        prep_data_config,
        combine_data_config,
        retry_times=retry_times)
    
    train_op = Train(
        "train",
        TrainModel,
        train_config,
        retry_times=retry_times)
    
    pflow_op = ProteinFlow(
        "pflow",
        cmd_op,
        select_op,
        label_op,
        data_op,
        train_op,
        workflow_steps_config
    )
    return pflow_op


def submit_pflow(
        confs: Union[str, List[str]],
        topology: Optional[str],
        pflow_config: str,
        machine_config: str,
        forcefield: Optional[str] = None,
        index_file: Optional[str] = None,
    ):
    with open(machine_config, "r") as mcg:
        machine_config_dict = json.load(mcg)
    resources = machine_config_dict["resources"]
    tasks = machine_config_dict["tasks"]
    normalized_resources = {}
    for resource_type in resources.keys():
        normalized_resources[resource_type] = normalize_resources(resources[resource_type])
        
    pflow_op = prep_pflow_op(
        prep_cmd_config = normalized_resources[tasks["prep_cmd_config"]],
        run_cmd_config = normalized_resources[tasks["run_cmd_config"]],
        prep_label_config = normalized_resources[tasks["prep_label_config"]],
        run_label_config = normalized_resources[tasks["run_label_config"]],
        run_select_config = normalized_resources[tasks["run_select_config"]],
        prep_data_config = normalized_resources[tasks["prep_data_config"]],
        combine_data_config = normalized_resources[tasks["combine_data_config"]],
        train_config = normalized_resources[tasks["train_config"]],
        workflow_steps_config = normalized_resources[tasks["workflow_steps_config"]],
        retry_times=1
    )

    if isinstance(confs, str):
        confs_artifact = upload_artifact(Path(confs), archive=None)
    elif isinstance(confs, List):
        confs_artifact = upload_artifact([Path(p) for p in confs], archive=None)
    else:
        raise RuntimeError("Invalid type of `confs`.")

    if index_file is None:
        index_file_artifact = None
    else:
        index_file_artifact = upload_artifact(Path(index_file), archive=None)
    
    jdata = deepcopy(load_json(pflow_config))
    
    if forcefield is None:
        forcefield_artifact = None
    else:
        forcefield_artifact = upload_artifact(Path(forcefield), archive=None)
        
    if topology is None:
        top_artifact = None
    else:
        top_artifact = upload_artifact(Path(topology), archive=None)
        
    pflow_config = upload_artifact(Path(pflow_config), archive=None)
        
    pflow_steps = Step("pflow-step",
            pflow_op,
            artifacts={
                "topology": top_artifact,
                "forcefield": forcefield_artifact,
                "confs": confs_artifact,
                "index_file": index_file_artifact,
                "pflow_config": pflow_config
            },
            parameters={},
        )
    wf = Workflow("pflow-workflow", pod_gc_strategy="OnPodSuccess", parallelism=50)
    wf.add(pflow_steps)
    wf.submit()
