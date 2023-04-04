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
from .submit import prep_pflow_op


def resubmit_pflow(
        workflow_id: str,
        confs: Union[str, List[str]],
        topology: Optional[str],
        pflow_config: str,
        machine_config: str,
        pod: Optional[str] = None,
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
        retry_times=None
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
    
    old_workflow = Workflow(id=workflow_id)
    all_steps = old_workflow.query_step()

    succeeded_steps = []
    restart_flag = 1
    for step in all_steps:
        if step["type"] == "Pod":
            # if step["phase"] == "Succeeded":
            if step["key"] != "prepare-pflow":
                pod_key = step["key"]
                if pod_key is not None:
                    pod_key_list = pod_key.split("-")
                    pod_step = "-".join(pod_key_list[:-1])
                    if pod is not None:
                        if pod_step == pod:
                            restart_flag = 0
                    else:
                        if step["phase"] != "Succeeded":
                            restart_flag = 0
                        else:
                            restart_flag = 1
                
                if restart_flag == 1:
                    succeeded_steps.append(step)
    wf = Workflow("pdflow-workflow-continue", pod_gc_strategy="OnPodSuccess", parallelism=50)
    wf.add(pflow_steps)
    wf.submit(reuse_step=succeeded_steps)