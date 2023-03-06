from typing import Dict, List
from copy import deepcopy
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


class CMD(Steps):
    
    r"""" cmd SuperOP.
    This SuperOP combines PrepExplore OP and RunExplore OP.
    """
    def __init__(
        self,
        name: str,
        prep_op: OP,
        run_op: OP,
        prep_config: Dict,
        run_config: Dict,
        upload_python_package = None,
        retry_times = None
    ):
        self._input_parameters = {
            "cmd_config" : InputParameter(type=Dict),
            "cmd_cv_config" : InputParameter(type=Dict),
            "task_names" : InputParameter(type=List[str])
        }        
        self._input_artifacts = {
            "forcefield": InputArtifact(optional=True),
            "topology" : InputArtifact(optional=True),
            "confs" : InputArtifact(),
            "index_file": InputArtifact(optional=True)
        }
        self._output_artifacts = {
            "plm_out": OutputArtifact(),
            "md_log": OutputArtifact(),
            "trajectory": OutputArtifact(),
            "conf_outs": OutputArtifact()
        }

        super().__init__(        
                name=name,
                inputs=Inputs(
                    parameters=self._input_parameters,
                    artifacts=self._input_artifacts
                ),
                outputs=Outputs(
                    artifacts=self._output_artifacts
                ),
            )

        step_keys = {
            "prep_cmd": "prep-cmd",
            "run_cmd": "run-cmd",
        }

        self = _cmd(
            self, 
            step_keys,
            prep_op,
            run_op,
            prep_config = prep_config,
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


def _cmd(
        cmd_steps,
        step_keys,
        prep_cmd_op : OP,
        run_cmd_op : OP,
        prep_config : Dict,
        run_config : Dict,
        upload_python_package : str = None,
        retry_times: int = None
    ):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop('template_config')
    run_template_config = run_config.pop('template_config')
    prep_executor = init_executor(prep_config.pop('executor'))
    run_executor = init_executor(run_config.pop('executor'))
    prep_cmd = Step(
        'prep-cmd',
        template=PythonOPTemplate(
            prep_cmd_op,
            python_packages = upload_python_package,
            retry_on_transient_error = retry_times,
            slices=Slices("{{item}}",
                input_parameter=["task_name"],
                input_artifact=["conf"],
                output_artifact=["task_path"]
            ),
            **prep_template_config,
        ),
        parameters={
            "cmd_config" : cmd_steps.inputs.parameters['cmd_config'],
            "cmd_cv_config" : cmd_steps.inputs.parameters['cmd_cv_config'],
            "task_name": cmd_steps.inputs.parameters['task_names']
        },
        artifacts={
            "topology" :cmd_steps.inputs.artifacts['topology'],
            "conf" : cmd_steps.inputs.artifacts['confs']
        },
        key = step_keys["prep_cmd"]+"-{{item}}",
        with_param=argo_range(argo_len(cmd_steps.inputs.parameters['task_names'])),
        executor = prep_executor,
        **prep_config,
    )
    cmd_steps.add(prep_cmd)

    run_cmd = Step(
        'run-cmd',
        template=PythonOPTemplate(
            run_cmd_op,
            python_packages = upload_python_package,
            retry_on_transient_error = retry_times,
            slices=Slices("{{item}}",
                input_artifact=["task_path"],
                output_artifact=["plm_out", "trajectory", "md_log", "conf_out"]
            ),
            **run_template_config,
        ),
        parameters={
            "cmd_config" : cmd_steps.inputs.parameters["cmd_config"]
        },
        artifacts={
            "task_path" : prep_cmd.outputs.artifacts["task_path"],
            "forcefield": cmd_steps.inputs.artifacts['forcefield'],
            "index_file": cmd_steps.inputs.artifacts['index_file']
        },
        key = step_keys["run_cmd"]+"-{{item}}",
        executor = run_executor,
        with_param=argo_range(argo_len(cmd_steps.inputs.parameters['task_names'])),
        **run_config,
    )
    cmd_steps.add(run_cmd)

    cmd_steps.outputs.artifacts["plm_out"]._from = run_cmd.outputs.artifacts["plm_out"]
    cmd_steps.outputs.artifacts["md_log"]._from = run_cmd.outputs.artifacts["md_log"]
    cmd_steps.outputs.artifacts["trajectory"]._from = run_cmd.outputs.artifacts["trajectory"]
    cmd_steps.outputs.artifacts["conf_outs"]._from = run_cmd.outputs.artifacts["conf_out"]
    
    return cmd_steps