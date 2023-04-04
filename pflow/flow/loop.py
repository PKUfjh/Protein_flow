from dflow import (
    InputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Step,
    Steps,
    if_expression,
)
from dflow.python import(
    PythonOPTemplate,
    OP
)
from typing import List, Optional, Dict, Union
import numpy as np
from pflow.utils import init_executor
from pflow.op.prep_pflow import PrepPflow
from copy import deepcopy


class ProteinFlow(Steps):
    def __init__(
            self,
            name : str,
            cmd_op : Steps,
            selector_op: Steps,
            label_op: Steps,
            data_op: Steps,
            train_op: Steps,
            step_config : dict,
            upload_python_package : str = None,
    ):
        
        self._input_parameters={}
        self._input_artifacts={
            "forcefield": InputArtifact(optional=True),
            "topology": InputArtifact(optional=True),
            "confs": InputArtifact(),
            "pflow_config": InputArtifact(),
            "index_file": InputArtifact(optional=True)
        }
        self._output_parameters={
        }
        self._output_artifacts={
            "trajectory": OutputArtifact(),
            "conf_outs": OutputArtifact()
        }        
        
        super().__init__(
            name = name,
            inputs = Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        _init_keys = ['recorder', 'block']
        step_keys = {}
        for ii in _init_keys:
            step_keys[ii] = '--'.join(['init', ii])

        self = _pflow(
            self,
            step_keys,
            name, 
            cmd_op,
            selector_op,
            label_op,
            data_op,
            train_op,
            step_config = step_config,
            upload_python_package = upload_python_package,
        )

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def init_keys(self):
        return self._init_keys

    @property
    def loop_keys(self):
        return [self.loop_key] + self.loop.keys


def _pflow(
        steps, 
        step_keys,
        name,
        cmd_op,
        selector_op,
        label_op,
        data_op,
        train_op,
        step_config : dict,
        upload_python_package : Optional[str] = None
    ):  

    _step_config = deepcopy(step_config)
    step_template_config = _step_config.pop('template_config')
    step_executor = init_executor(_step_config.pop('executor'))

    prep_pflow = Step(
        name = 'prepare-pflow',
        template=PythonOPTemplate(
            PrepPflow,
            python_packages = upload_python_package,
            **step_template_config,
        ),
        parameters={},
        artifacts={
            "confs": steps.inputs.artifacts['confs'],
            "pflow_config" : steps.inputs.artifacts['pflow_config'],
        },
        key = 'prepare-pflow',
        executor = step_executor,
        **_step_config,
    )
    steps.add(prep_pflow)

    cmd_pflow = Step(
        name = 'cmd-block',
        template = cmd_op,
        parameters={
            "task_names": prep_pflow.outputs.parameters["walker_tags"],
            "cmd_config": prep_pflow.outputs.parameters["cmd_config"],
            "cmd_cv_config": prep_pflow.outputs.parameters["cmd_cv_config"],
        },
        artifacts={
            "forcefield" : steps.inputs.artifacts['forcefield'],
            "topology": steps.inputs.artifacts["topology"],
            "confs": prep_pflow.outputs.artifacts["confs"],
            "index_file": steps.inputs.artifacts["index_file"],
        },
        key = "cmd-block"
    )
    steps.add(cmd_pflow)
    
    select_pflow = Step(
        name = 'select-block',
        template = selector_op,
        parameters={
            "cluster_threshold": prep_pflow.outputs.parameters["cluster_threshold"],
            "angular_mask": prep_pflow.outputs.parameters["angular_mask"],
            "weights": prep_pflow.outputs.parameters["weights"],
            "numb_cluster_upper": prep_pflow.outputs.parameters['numb_cluster_upper'],
            "numb_cluster_lower": prep_pflow.outputs.parameters['numb_cluster_lower'],
            "max_selection": prep_pflow.outputs.parameters["max_selection"],
            "dt": prep_pflow.outputs.parameters["dt"],
            "output_freq": prep_pflow.outputs.parameters["output_freq"],
            "slice_mode": prep_pflow.outputs.parameters["slice_mode"],
            "task_names": prep_pflow.outputs.parameters["walker_tags"],
        },
        artifacts={
            "plm_out": cmd_pflow.outputs.artifacts["plm_out"],
            "xtc_traj": cmd_pflow.outputs.artifacts["trajectory"],
            "topology": prep_pflow.outputs.artifacts["confs"],
        },
        key = "select-block"
    )
    steps.add(select_pflow)
    
    label_pflow =  Step(
        "Label",
        template=label_op,
        parameters={
            "label_config": prep_pflow.outputs.parameters["label_config"],
            "label_cv_config": prep_pflow.outputs.parameters["label_cv_config"],
            "conf_tags" : select_pflow.outputs.parameters['selected_conf_tags']
        },
        artifacts={
            "topology": steps.inputs.artifacts["topology"],
            "forcefield" : steps.inputs.artifacts['forcefield'],
            "confs": select_pflow.outputs.artifacts["selected_confs"],
            "index_file": steps.inputs.artifacts["index_file"]
        },
        key = 'label-block'
    )
    steps.add(label_pflow)
    
    data_pflow =  Step(
        "Data",
        template=data_op,
        parameters={
            "task_name": label_pflow.outputs.parameters['conf_tags']
        },
        artifacts={
            "conf_begin": label_pflow.outputs.artifacts['conf_begin'],
            "trajectory_aligned" : label_pflow.outputs.artifacts['trajectory_aligned']
        },
        key = 'data-block'
    )
    steps.add(data_pflow)
    
    train_pflow =  Step(
    "Train",
    template=train_op,
    parameters={
        "model_tags": prep_pflow.outputs.parameters['model_tags'],
        "train_config": prep_pflow.outputs.parameters["train_config"]
    },
    artifacts={
        "data": data_pflow.outputs.artifacts["combined_npz"]
    },
    key = 'train-block'
    )
    steps.add(train_pflow)
    
    steps.outputs.artifacts['trajectory']._from = cmd_pflow.outputs.artifacts['trajectory']
    steps.outputs.artifacts['conf_outs']._from =  cmd_pflow.outputs.artifacts['conf_outs']
    
    return steps