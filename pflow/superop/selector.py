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


class Selector(Steps):
    
    r""" Selector SuperOP.
    This SuperOP combines PrepSelect OP and RunSelect OP.    
    """
    def __init__(
        self,
        name: str,
        run_op: OP,
        run_config: Dict,
        upload_python_package = None,
        retry_times = None
    ):
        self._input_parameters = {
            "cluster_threshold": InputParameter(type=List[float], value=1.0),
            "angular_mask": InputParameter(type=Optional[Union[np.ndarray, List]]),
            "weights": InputParameter(type=Optional[Union[np.ndarray, List]]),
            "numb_cluster_upper": InputParameter(type=Optional[float], value=None),
            "numb_cluster_lower": InputParameter(type=Optional[float], value=None),
            "max_selection": InputParameter(type=int),
            "dt": InputParameter(type=float, value=0.02),
            "output_freq": InputParameter(type=float, value=2500),
            "slice_mode": InputParameter(type=str, value="gmx"),
            "task_names" : InputParameter(type=List[str])
        }        
        self._input_artifacts = {
            "plm_out": InputArtifact(),
            "xtc_traj": InputArtifact(),
            "topology": InputArtifact()
        }
        self._output_parameters = {
            "numb_cluster": OutputParameter(type=List[int]),
            "selected_conf_tags": OutputParameter(type=List),
        }
        self._output_artifacts = {
            "selected_confs": OutputArtifact(),
            "selected_indices": OutputArtifact()
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
            "run_select": "run-select"
        }

        self = _select(
            self, 
            step_keys,
            run_op,
            run_config = run_config,
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


def _select(
        select_steps,
        step_keys,
        run_select_op : OP,
        run_config : Dict,
        upload_python_package : str = None,
        retry_times: int = None
    ):
    run_config = deepcopy(run_config)
    run_template_config = run_config.pop('template_config')
    run_executor = init_executor(run_config.pop('executor'))

    run_select = Step(
        'run-select',
        template=PythonOPTemplate(
            run_select_op,
            python_packages = upload_python_package,
            retry_on_transient_error = retry_times,
            slices=Slices(sub_path = True,
                input_parameter=["cluster_threshold", "task_name"],
                input_artifact=["plm_out","xtc_traj","topology"],
                output_artifact=["selected_confs", "selected_indices"],
                output_parameter=["numb_cluster"]
            ),
            **run_template_config,
        ),
        parameters={
            "cluster_threshold": select_steps.inputs.parameters['cluster_threshold'],
            "angular_mask": select_steps.inputs.parameters['angular_mask'],
            "weights": select_steps.inputs.parameters['weights'],
            "numb_cluster_upper": select_steps.inputs.parameters['numb_cluster_upper'],
            "numb_cluster_lower": select_steps.inputs.parameters['numb_cluster_lower'],
            "max_selection": select_steps.inputs.parameters['max_selection'],
            "dt": select_steps.inputs.parameters['dt'],
            "output_freq": select_steps.inputs.parameters['output_freq'],
            "slice_mode": select_steps.inputs.parameters["slice_mode"],
            "task_name": select_steps.inputs.parameters['task_names']
        },
        artifacts={
            "plm_out": select_steps.inputs.artifacts['plm_out'],
            "xtc_traj": select_steps.inputs.artifacts['xtc_traj'],
            "topology": select_steps.inputs.artifacts['topology'],
        },
        key = step_keys["run_select"]+"-{{item.order}}",
        executor = run_executor,
        **run_config,
    )
    select_steps.add(run_select)

    select_steps.outputs.parameters["numb_cluster"].value_from_parameter = run_select.outputs.parameters["numb_cluster"]
    select_steps.outputs.parameters["selected_conf_tags"].value_from_parameter = run_select.outputs.parameters["selected_conf_tags"]

    select_steps.outputs.artifacts["selected_confs"]._from = run_select.outputs.artifacts["selected_confs"]
    select_steps.outputs.artifacts["selected_indices"]._from = run_select.outputs.artifacts["selected_indices"]
    
    
    
    return select_steps
