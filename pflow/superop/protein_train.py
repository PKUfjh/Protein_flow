from typing import Dict, List, Optional, Union
from copy import deepcopy
import numpy as np
from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Step,
    Steps,
    argo_range,
    argo_len,
)
from dflow.python import(
    PythonOPTemplate,
    OP,
    Slices
)
from pflow.utils import init_executor


class Train(Steps):
    
    r"""Train SuperOP.
    """
    def __init__(
        self,
        name: str,
        train_op: OP,
        train_config: Dict,
        upload_python_package = None,
        retry_times = None
    ):
        self._input_parameters = {
            "model_tags":  InputParameter(type=List[str]),
            "train_config": InputParameter(type=Dict)
        }        
        self._input_artifacts = {
            "data": InputArtifact()
        }
        self._output_parameters = {}
        self._output_artifacts = {
            "model_log": OutputArtifact()
        }

        super().__init__(        
                name=name,
                inputs=Inputs(
                    parameters=self._input_parameters,
                    artifacts=self._input_artifacts
                ),
                outputs=Outputs(
                    parameters=self._output_parameters,
                    artifacts=self._output_artifacts
                ),
            )
        
        step_keys = {
            "run_train": "run-train"
        }

        self = _train(
            self, 
            step_keys,
            train_op,
            train_config = train_config,
            upload_python_package = upload_python_package,
            retry_times = retry_times
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
    def keys(self):
        return self._keys


def _train(
        train_steps,
        step_keys,
        run_train_op : OP,
        train_config : Dict,
        upload_python_package : str = None,
        retry_times: int = None
    ):
    train_config = deepcopy(train_config)
    run_template_config = train_config.pop('template_config')
    run_executor = init_executor(train_config.pop('executor'))

    run_train = Step(
        'run-train',
        template=PythonOPTemplate(
            run_train_op,
            python_packages = upload_python_package,
            retry_on_transient_error = retry_times,
            slices=Slices("{{item}}",
                input_parameter=["model_tags"],
                output_artifact=["model_log"]
            ),
            **run_template_config,
        ),
        parameters={
            "model_tags": train_steps.inputs.parameters["model_tags"],
            "train_config": train_steps.inputs.parameters["train_config"]
        },
        artifacts={
            "data": train_steps.inputs.artifacts["data"],
        },
        with_param=argo_range(argo_len(train_steps.inputs.parameters["model_tags"])),
        key = step_keys["run_train"]+"-{{item}}",
        executor = run_executor,
        **train_config,
    )
    train_steps.add(run_train)
    train_steps.outputs.artifacts["model_log"]._from = run_train.outputs.artifacts["model_log"]
    
    return train_steps
